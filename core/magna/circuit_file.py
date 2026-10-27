"""Source-located Stim-style files and a purely static classical control DAG.

Parsing never imports a protocol, executes a controller, or simulates a state.
Records are absolute reported-record indices and control writes have SSA versions.
The original text is retained for lossless serialization; flattened instructions
carry repeat addresses so physical sites remain distinct after expansion.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from . import _native


_BUILTIN_CONTROLS = frozenset(("XOR", "NOT"))


@dataclass(frozen=True)
class SourceLocation:
    filename: str
    line: int
    column: int = 1
    repeat: tuple = ()

    def __str__(self):
        address = "" if not self.repeat else " (repeat " + ".".join(map(str, self.repeat)) + ")"
        return f"{self.filename}:{self.line}:{self.column}{address}"


class CircuitSyntaxError(ValueError):
    def __init__(self, message, source):
        self.source = source
        super().__init__(f"{source}: {message}")


@dataclass(frozen=True, order=True)
class Target:
    kind: str
    index: int
    version: int = -1
    inverted: bool = False
    pauli: str = ""

    @property
    def key(self):
        return (self.kind, self.index, self.version)


@dataclass(frozen=True)
class AffineExpr:
    constant: int = 0
    terms: frozenset = frozenset()

    def _raw(self):
        return self.constant, tuple(_target_data(t) for t in sorted(self.terms))


@dataclass(frozen=True)
class Instruction:
    name: str
    args: tuple
    targets: tuple
    source: SourceLocation
    op_id: str
    outputs: tuple = ()
    records: tuple = ()
    control_name: str | None = None
    guard: Target | None = None
    products: tuple = ()
    chain_id: str | None = None


def _controller_error(message, source):
    return ValueError(message if source is None else f"{source}: {message}")


@dataclass(frozen=True)
class ControllerBinding:
    name: str
    version: str
    callback: object = field(repr=False, compare=False)
    config: object = field(default_factory=dict)

    def __post_init__(self):
        if self.name in _BUILTIN_CONTROLS or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", self.name):
            raise ValueError("invalid or reserved controller name")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("controller version must be a nonempty string")
        if not callable(self.callback):
            raise TypeError("controller callback must be callable")
        # A canonical JSON snapshot makes caller mutations irrelevant.
        canonical = json.dumps(self.config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "_config_json", canonical)
        object.__setattr__(self, "config", json.loads(canonical))

    @property
    def fingerprint(self):
        return sha256(f"{self.name}\n{self.version}\n{self._config_json}".encode()).hexdigest()

    def invoke(self, bits, output_count, source=None):
        """Call the installed decoder on ordered bits and a fresh JSON config."""
        inputs = tuple(bits)
        if not _native.controller_bits_valid(inputs):
            raise _controller_error(f"controller {self.name} inputs must be bits", source)
        try:
            # The common empty configuration still gets a fresh dict each call.
            config = {} if self._config_json == "{}" else json.loads(self._config_json)
            result = tuple(self.callback(inputs, config))
        except Exception as exc:
            raise _controller_error(f"controller {self.name}@{self.version} failed: {exc}", source) from exc
        if len(result) != output_count:
            raise _controller_error(
                f"controller {self.name} returned {len(result)} bits; expected {output_count}", source)
        if not _native.controller_bits_valid(result):
            raise _controller_error(f"controller {self.name} returned a non-bit", source)
        return tuple(map(int, result))


class ControllerRegistry:
    """Explicit installed-controller bindings; text never names Python imports."""
    def __init__(self, bindings=()):
        self._bindings = {}
        source = bindings.values() if isinstance(bindings, dict) else bindings
        for binding in source:
            if not isinstance(binding, ControllerBinding):
                raise TypeError("registry entries must be ControllerBinding objects")
            self.register(binding.name, binding.callback, version=binding.version,
                          config=json.loads(binding._config_json))

    def register(self, name, callback, *, version, config=None):
        if name in self._bindings:
            raise ValueError(f"controller {name!r} is already registered")
        binding = ControllerBinding(name, version, callback, {} if config is None else config)
        self._bindings[name] = binding
        return binding

    def resolve(self, name):
        try:
            return self._bindings[name]
        except KeyError as exc:
            raise ValueError(f"missing controller binding {name!r}; install and explicitly register its decoder") from exc

    def snapshot(self):
        return ControllerRegistry(tuple(self._bindings.values()))

    @property
    def bindings(self):
        return MappingProxyType(self._bindings)

    def to_dict(self):
        return {name: {"version": b.version, "config": json.loads(b._config_json)}
                for name, b in sorted(self._bindings.items())}


@dataclass(frozen=True)
class ControlAnalysis:
    route: str
    affine: object
    required_parities: tuple
    external_instructions: tuple
    reasons: tuple
    detectors: object = field(default_factory=dict)
    _native_handle: object = field(default=None, init=False, repr=False, compare=False)

    def expression(self, target):
        if self._native_handle is not None:
            return _expression_value(_native.file_expression('expression', self._native_handle, _target_data(target)))
        # Adapter for explicitly constructed public analysis values.
        return _expression_value(_native.file_expression('resolve', _target_data(target),
            tuple((_target_data(t), e._raw()) for t, e in self.affine.items()),
            tuple(e._raw() for e in self.detectors.values())))


@dataclass(frozen=True)
class CircuitFile:
    instructions: tuple
    num_records: int
    num_qubits: int
    analysis: ControlAnalysis
    controllers: ControllerRegistry
    source: str
    text: str = field(repr=False)
    _native_handle: object = field(default=None, init=False, repr=False, compare=False)

    def to_text(self):
        return self.text

    def write(self, path):
        Path(path).write_text(self.text, encoding="utf-8")

    @property
    def format_version(self):
        return _native.file_expression('format', _circuit_handle(self))

    @property
    def fingerprint(self):
        # Include actual binding identity declarations, not callback addresses.
        bindings = {i.control_name: self.controllers.resolve(i.control_name).fingerprint
                    for i in self.analysis.external_instructions}
        return sha256(json.dumps([self.format_version, self.text, bindings], sort_keys=True).encode()).hexdigest()

    def to_dict(self):
        def target(t):
            return dict(kind=t.kind, index=t.index, version=t.version,
                        inverted=t.inverted, pauli=t.pauli)
        def expression(expr):
            return {"constant": expr.constant, "terms": [target(t) for t in sorted(expr.terms)]}
        return {"format": self.format_version, "num_qubits": self.num_qubits,
                "num_records": self.num_records, "route": self.analysis.route,
                "control_analysis": {
                    "reasons": list(self.analysis.reasons),
                    "required_parities": [expression(e) for e in self.analysis.required_parities],
                    "detectors": [expression(e) for e in self.analysis.detectors.values()],
                    "control_writes": [{"target": target(t), "expression": expression(e)}
                                       for t, e in sorted(self.analysis.affine.items())],
                    "external_inputs": [{"op_id": i.op_id, "controller": i.control_name,
                                         "source": str(i.source),
                                         "inputs": [expression(self.analysis.expression(t)) for t in i.targets]}
                                        for i in self.analysis.external_instructions]},
                "fingerprint": self.fingerprint,
                "controllers": {name: {"version": b.version, "config": json.loads(b._config_json)}
                                for name, b in self.controllers.bindings.items()},
                "instructions": [dict(name=i.name, args=[str(x) for x in i.args],
                    targets=[target(t) for t in i.targets], outputs=[target(t) for t in i.outputs],
                    records=list(i.records), control_name=i.control_name,
                    guard=None if i.guard is None else target(i.guard),
                    products=[[target(t) for t in p] for p in i.products],
                    chain_id=i.chain_id, op_id=i.op_id,
                    source=dict(filename=i.source.filename, line=i.source.line,
                                column=i.source.column, repeat=list(i.source.repeat)))
                    for i in self.instructions]}


def _target_data(target):
    return target.kind, target.index, target.version, target.inverted, target.pauli


def _circuit_handle(circuit):
    if circuit._native_handle is None:
        raise ValueError('this CircuitFile has no parsed form; build circuits with parse_circuit or load_circuit '
                         'rather than constructing or replacing CircuitFile fields')
    return circuit._native_handle


def _expression_value(raw):
    return AffineExpr(raw[0], frozenset(Target(*t) for t in raw[1]))


def parse_circuit(text, *, controllers=None, source="<string>", max_instructions=1_000_000, max_qubits=65_536):
    """Parse and validate natively; callbacks are bound but never executed here."""
    if not isinstance(text, str):
        raise TypeError("circuit text must be a string")
    for name, value in (("max_instructions", max_instructions), ("max_qubits", max_qubits)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    registry = controllers.snapshot() if isinstance(controllers, ControllerRegistry) else ControllerRegistry(controllers or ())
    handle, raw = _native.parse_file(text, str(source), tuple(registry.bindings), max_instructions, max_qubits)
    if handle is None:
        raise CircuitSyntaxError(raw['error'], SourceLocation(*raw['source']))
    return _circuit_from_native((handle, raw), registry)


def _circuit_from_native(snapshot, registry):
    handle, raw = snapshot
    instructions = tuple(Instruction(name, tuple(Fraction(*p) for p in args), tuple(Target(*t) for t in targets),
        SourceLocation(*location), op_id, tuple(Target(*t) for t in outputs), records, control_name,
        None if guard is None else Target(*guard), tuple(tuple(Target(*t) for t in group) for group in products), chain)
        for name, args, targets, location, op_id, outputs, records, control_name, guard, products, chain in raw['instructions'])
    external = tuple(instructions[j] for j in raw['external'])
    analysis = ControlAnalysis('nonlinear' if external else 'linear',
        MappingProxyType({Target(*t): _expression_value(e) for t, e in raw['affine']}),
        tuple(_expression_value(e) for e in raw['required']), external,
        ('external CONTROL functions require actual decoder evaluation',) if external
        else ('all CONTROL functions are built-in XOR or NOT',),
        MappingProxyType({j: _expression_value(e) for j, e in enumerate(raw['detectors'])}))
    object.__setattr__(analysis, '_native_handle', handle)
    circuit = CircuitFile(instructions, raw['num_records'], raw['num_qubits'], analysis, registry, raw['source'], raw['text'])
    object.__setattr__(circuit, '_native_handle', handle)
    return circuit


def load_circuit(path, *, controllers=None, max_instructions=1_000_000, max_qubits=65_536):
    path = Path(path)
    return parse_circuit(path.read_text(encoding="utf-8"), controllers=controllers,
                         source=str(path), max_instructions=max_instructions, max_qubits=max_qubits)
