"""Load circuit files, compile tasks, run them, and the magna command line.

Compilation, evaluation and sampling execute in the native kernel. This module
checks the user's task, options and sampling configuration, binds explicitly
registered decoders, and turns native results into JSON-ready dictionaries.
"""
from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
from fractions import Fraction
from pathlib import Path
from random import Random
from time import monotonic
import argparse
import importlib
import json
import math
import sys

from ._native import ResourceLimit
from .circuit_file import load_circuit
from .ir import fingerprint

SCHEMA_VERSION = 2
PACKAGE_VERSION = '0.1.0'
NATIVE_LIMITS = {'max_path_bits', 'max_doubled_path_bits', 'max_quotient_rank',
                 'max_terms', 'max_table_bytes'}
RUN_LIMITS = {'max_plans', 'max_seconds', 'max_queries'}
HISTORY_SAMPLERS = ('affine_support', 'exact_marginal')
TASK_FIELDS = {'schema_version', 'outputs', 'observables', 'error_observables', 'sample_records',
               'ideal_state', 'recovery'}
_ACCEPTANCE = 'declare acceptance in the circuit with POSTSELECT det[n] (!det[n] requires a one)'
REMOVED_TASK_FIELDS = {
    'detectors': _ACCEPTANCE,
    'postselection': _ACCEPTANCE+', after a DETECTOR for each parity',
    'outcomes': _ACCEPTANCE,
    'error_checks': 'declare each parity with OBSERVABLE_INCLUDE(k) in the circuit and list k in error_observables',
    'decoded_error': 'pass detectors to CONTROL[decoder], add its output to an OBSERVABLE_INCLUDE '
                     'and list that observable in error_observables',
}


def _unknown(kind, names, allowed):
    """Name every unrecognised key, suggest the closest valid one and list the valid keys."""
    parts = []
    for name in sorted(map(str, names)):
        close = get_close_matches(name, sorted(allowed), n=1)
        parts.append(repr(name) + (f' (did you mean {close[0]!r}?)' if close else ''))
    return ValueError(f"unknown {kind}: {', '.join(parts)}. Valid: {', '.join(sorted(allowed))}")


def _read_json(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text())
    return deepcopy(dict(value or {}))


def load_controllers(path):
    """Load explicitly authorized installed Python bindings, never circuit code."""
    from .circuit_file import ControllerRegistry
    data = _read_json(path)
    if data.get('schema_version', 1) != 1:
        raise ValueError('unsupported controller schema_version')
    registry = ControllerRegistry()
    for name, spec in data.get('bindings', {}).items():
        module, sep, function = spec['import'].partition(':')
        if not sep or not module or not function or '.' in function:
            raise ValueError('controller import must be package.module:function')
        try:
            callback = getattr(importlib.import_module(module), function)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f'controller binding {name!r} cannot import {spec["import"]!r}: {exc}') from exc
        registry.register(name, callback, version=spec['version'], config=spec.get('config', {}))
    return registry


@dataclass(frozen=True)
class NoiseDraw:
    gates: dict
    report_flips: tuple
    likelihood_ratio: tuple = (1, 1)


def _sampling(supplied):
    data = _read_json(supplied)
    method = data.get('method', 'physical')
    if method == 'physical':
        if set(data)-{'method'}:
            raise ValueError('physical sampling takes no other fields; remove '+', '.join(sorted(set(data)-{'method'})))
        return dict(method='physical')
    if method != 'importance':
        raise ValueError(f"sampling method must be 'physical' or 'importance', not {method!r}")
    if set(data)-{'method', 'proposal_scale'}:
        raise _unknown('importance sampling fields', set(data)-{'method', 'proposal_scale'}, {'method', 'proposal_scale'})
    scale = data.get('proposal_scale')
    if type(scale) not in (str, int, float, Fraction):
        raise ValueError('importance sampling requires a rational proposal_scale, such as "10" or "500/77"')
    try: scale = Fraction(str(scale))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'proposal_scale must be a finite positive rational, not {data["proposal_scale"]!r}') from exc
    if scale <= 0: raise ValueError(f'proposal_scale must be positive, not {data["proposal_scale"]!r}')
    return dict(method='importance', proposal_scale=str(scale))


class _NoiseSchedule:
    """Immutable original channel categories; repeated draws are native."""
    def __init__(self, circuit, sampling=None):
        self.sampling = _sampling(sampling)
        from . import _native
        from .circuit_file import _circuit_handle
        scale = Fraction(self.sampling['proposal_scale']) if self.sampling['method'] == 'importance' else None
        self._native_plan, metadata = _native.compile_file_noise(_circuit_handle(circuit),
            None if scale is None else (scale.numerator, scale.denominator))
        self.category_count = metadata['categories']
        self.action_count = metadata['actions']
        self.tilted = metadata['tilted']
        self.guarded = metadata['guarded']
        self.deterministic = metadata['deterministic']

    @property
    def probabilities(self):
        from . import _native
        return _native.file_noise_probabilities(self._native_plan)

    @staticmethod
    def _gates(value):
        return tuple(dict(kind=p, qubits=[q]) for p, q in value)

    @property
    def actions(self):
        # Enumeration and explicit inspection retain their Python metadata view.
        # Ordinary sampling never exports or retains the complete action table.
        from . import _native
        if self._native_plan is None: return ()
        return tuple(tuple((oid, value if oid == 'report' else self._gates(value))
                           for oid, value in group)
                     for group in _native.file_noise_actions(self._native_plan))

    def to_dict(self, *, history_sampler=None):
        return dict(self.sampling, categories=self.category_count, tilted_categories=self.tilted,
                    guarded_categories_unchanged=self.guarded, deterministic_categories_unchanged=self.deterministic,
                    exact_marginal_records='true conditional distribution' if history_sampler == 'exact_marginal' else None,
                    channel_proportions='unchanged')


def _check_task_fields(task):
    for name, guidance in REMOVED_TASK_FIELDS.items():
        if name in task: raise ValueError(f"task field {name!r} is no longer supported; {guidance}")
    unknown = set(task)-TASK_FIELDS
    if unknown: raise _unknown('task fields', unknown, TASK_FIELDS)


def _task(circuit, supplied):
    task = _read_json(supplied)
    _check_task_fields(task)
    if task.get('schema_version', 1) != 1: raise ValueError('unsupported task schema_version')
    outputs = tuple(task.get('outputs', range(circuit.num_qubits)))
    bad = [q for q in outputs if type(q) is not int or not 0 <= q < circuit.num_qubits]
    if bad or len(set(outputs)) != len(outputs):
        raise ValueError(f'outputs must be distinct qubit indices in 0..{circuit.num_qubits-1}; '
                         + (f'invalid: {bad}' if bad else f'repeated in {list(outputs)}'))
    from . import _native
    from .circuit_file import _circuit_handle
    def rows(data):
        return _native.task_record_rows(circuit.num_records, tuple(data))
    post, rhs = _native.expand_file_task(_circuit_handle(circuit))
    observables = tuple(tuple(pair) for pair in task.get('observables', ()))
    for pair in observables:
        if len(pair) != 2 or any(type(v) is not int or v < 0 or v >> len(outputs) for v in pair):
            raise ValueError(f'observables must be [x, z] bit masks on the {len(outputs)} output qubits; got {list(pair)}')
    samples = tuple(task.get('sample_records', ()))
    rows((r,) for r in samples)
    return dict(raw=task, outputs=outputs, postselection=post, outcomes=tuple(rhs),
                observables=observables, sample_records=samples)


def _options(supplied):
    options = dict(supplied or {})
    options.setdefault('history_sampler', 'affine_support')
    unknown = set(options) - NATIVE_LIMITS - RUN_LIMITS - {'history_sampler'}
    if unknown: raise _unknown('options', unknown, NATIVE_LIMITS | RUN_LIMITS | {'history_sampler'})
    for key, value in options.items():
        if key == 'history_sampler':
            if value not in HISTORY_SAMPLERS:
                raise ValueError(f'history_sampler must be affine_support or exact_marginal, not {value!r}')
        elif key == 'max_seconds':
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'max_seconds must be a finite positive number, not {value!r}')
        elif type(value) is not int or value < (1 if key in ('max_plans','max_queries') else 0):
            minimum = 'positive' if key in ('max_plans', 'max_queries') else 'nonnegative'
            raise ValueError(f'{key} must be a {minimum} integer, not {value!r}')
    return options


class CompiledCircuit:
    def __init__(self, circuit, *, task=None, options=None, ideal_state=None, recovery=None, controllers=None, sampling=None, _prepare=True):
        raw_task = _read_json(task)
        _check_task_fields(raw_task)
        # Reports describe the user's circuit; an error_observables task executes an internal extension of it.
        self._source_circuit = circuit
        self._error_observables = None
        record_target = None
        if 'error_observables' in raw_task:
            if (ideal_state is not None or recovery is not None or
                    any(key in raw_task for key in ('ideal_state', 'recovery')) or
                    raw_task.get('outputs', ()) not in ([], ()) or
                    raw_task.get('observables', ()) not in ([], ())):
                raise ValueError('error_observables asks a classical question: omit outputs and observables '
                                 'or leave them empty, and give no ideal_state or recovery')
            observables = raw_task['error_observables']
            if not isinstance(observables, (list, tuple)) or any(type(k) is not int or k < 0 for k in observables):
                raise ValueError(f'error_observables must list OBSERVABLE_INCLUDE indices, not {observables!r}')
            from . import _native
            from .circuit_file import _circuit_handle, _circuit_from_native
            self._error_observables = tuple(observables)
            physical, outputs, target = _native.lower_record_success(_circuit_handle(circuit), tuple(observables))
            circuit = _circuit_from_native(physical, circuit.controllers)
            record_target = _circuit_from_native(target, circuit.controllers)
            lowered_task = {key: value for key, value in raw_task.items() if key != 'error_observables'}
            lowered_task['outputs'] = outputs
        else:
            lowered_task = raw_task
        self.circuit = circuit
        self.task = _task(circuit, lowered_task)
        self.task['raw'] = raw_task
        from .ideal_state import IdealState
        from .terminal_recovery import TerminalRecovery
        target_spec = {}
        def module_file(name, supplied):
            nonlocal target_spec
            if supplied is not None and name in self.task['raw']:
                raise ValueError(name+' supplied both as an argument and in the task')
            value = supplied if supplied is not None else self.task['raw'].get(name)
            if value is None:
                return None
            if name == 'ideal_state' and isinstance(value, dict):
                if set(value)-{'circuit', 'outputs', 'postselection'} or 'circuit' not in value or 'outputs' not in value:
                    raise ValueError('target specification requires circuit and outputs, with optional auxiliary postselection')
                target_spec = {k: v for k, v in value.items() if k != 'circuit'}
                value = value['circuit']
            if isinstance(value, (str, Path)):
                path = Path(value)
                if supplied is None and isinstance(task, (str, Path)) and not path.is_absolute():
                    path = Path(task).parent/path
                return load_circuit(path, controllers=circuit.controllers if controllers is None else controllers)
            from .circuit_file import CircuitFile
            if not isinstance(value, CircuitFile):
                raise TypeError(name+' must be a CircuitFile or a file path')
            return value
        ideal_file = record_target if record_target is not None else module_file('ideal_state', ideal_state)
        recovery_file = module_file('recovery', recovery)
        if recovery_file is not None and ideal_file is None:
            raise ValueError('terminal recovery requires an ideal_state target')
        width = len(self.task['outputs'])
        self.ideal_state = None if ideal_file is None else IdealState(ideal_file, width, **target_spec)
        self.recovery = None if recovery_file is None else TerminalRecovery(recovery_file, width)
        self.options = _options(options)
        for inst in circuit.analysis.external_instructions:
            circuit.controllers.resolve(inst.control_name)
        self.noise_schedule = _NoiseSchedule(circuit, sampling)
        from .control_metadata import ControlMetadata
        self._control_metadata = ControlMetadata(circuit)
        self._recovery_control_metadata = None if self.recovery is None else ControlMetadata(self.recovery.circuit)
        self.queries = 0; self.compilations = 0
        self.kernel_seconds = 0.; self.compile_seconds = 0.; self.decoder_seconds = 0.
        self.started = monotonic()
        self.plans = (); self._plan_structures = ()
        from .native_file_runtime import NativeFileExecution
        self._execution = NativeFileExecution(self)
        self.route = self._execution.state(False)['route']
        self._execution_options = dict(self.options)
        self._compilation = dict(status='validated', query_families={})
        if _prepare:
            try: self._prepare_task()
            except (RuntimeError, ValueError, ArithmeticError, MemoryError) as exc:
                self._compilation['status'] = 'failed'
                exc.compiled_task = self
                raise

    @property
    def target_records(self):
        """Record names of the ideal-target projection, as the kernel assigns them."""
        return self._execution.state()['target_records']

    def _sync_execution(self, *, details=False):
        state = self._execution.state(details)
        for name in ('queries', 'compilations', 'kernel_seconds', 'compile_seconds', 'decoder_seconds'):
            setattr(self, name, state[name])
        self.plans = range(state['plans']); self._plan_structures = range(state['structures'])
        return state

    def _configure_execution(self):
        if self.options != self._execution_options:
            self._execution.configure(self.options)
            self._execution_options = dict(self.options)
            self._compilation['status'] = 'configuration_changed'

    def _prepare_task(self):
        tick = monotonic()
        try:
            self._execution.prepare()
        finally:
            state = self._sync_execution(details=True)
        self._compilation = dict(status='complete', query_families=state['query_families'],
            compiler_diagnostics=state['compiler_diagnostics'],
            resource_summary=state['resource_summary'],
            preparation_seconds=monotonic()-tick,
            conditional_noise=state['conditional_noise'], sampled_history=state['sampled_history'],
            guarded_noise_coverage='fixed reference networks with causal Pauli assignments' if self.noise_schedule.guarded else 'fixed template',
            reference_scope='declared boundary families; compatible reference outcomes are listed per query')

    def inspect(self):
        result = self._source_circuit.to_dict()
        result['task'] = self.task['raw']; result['options'] = self.options
        result['route'] = self.route
        result['execution'] = 'whole-instrument native boundary queries'
        result['evaluation'] = self.evaluation()
        result['sampling'] = self.noise_schedule.to_dict(history_sampler=self.options['history_sampler'])
        result['compilation'] = deepcopy(self._compilation)
        return result

    def evaluation(self):
        if self._error_observables is not None:
            return dict(ideal_state=None, recovery=None, output_mapping=[],
                        error_metric=f'error_observables {list(self._error_observables)}: '
                                     'wrong when any listed OBSERVABLE_INCLUDE parity is one',
                        observables='physical output')
        return dict(ideal_state=None if self.ideal_state is None else self.ideal_state.to_dict(),
                    recovery=None if self.recovery is None else self.recovery.to_dict(),
                    output_mapping=list(self.task['outputs']),
                    error_metric='target infidelity after terminal recovery' if self.ideal_state else 'declared output task',
                    observables='after recovery, before target projection' if self.recovery else 'physical output')

    def _history_result(self, result):
        masses = result['masses']
        def exact(raw):
            return None if raw is None else dict(coefficients=list(raw[0]), power=raw[1])
        row = dict(result['statistics'], sampled_records=list(result['samples']),
            decoder_calls=result['decoder_calls'], recovery_decoder_calls=result['recovery_decoder_calls'],
            sampled_parities=result['sampled_parities'], exact=dict(history_mass=exact(masses['history']),
                acceptance=exact(masses['acceptance']), error=exact(masses['error']),
                moments=[exact(v) for v in masses['moments']], target_success=exact(masses['target']),
                noise_likelihood_ratio_hex=[hex(v) for v in result['weight']]))
        if masses['proposal'] is not None:
            row['exact']['proposal_probability'] = dict(numerator=masses['proposal'][0], denominator=masses['proposal'][1])
        if masses['target_normalization'] != ((1, 0, 0, 0), 0):
            row['exact']['target_normalization'] = exact(masses['target_normalization'])
        return row

    def sample(self, rng, *, noise=None):
        self._configure_execution()
        try:
            return self._history_result(self._execution.sample(rng, noise))
        finally:
            self._sync_execution()


def compile_circuit(circuit, *, task=None, options=None, controllers=None, ideal_state=None, recovery=None, sampling=None):
    if isinstance(circuit, CompiledCircuit):
        raise TypeError('this circuit is already compiled; pass it to run(), or compile the original circuit again')
    if isinstance(circuit, (str, Path)): circuit = load_circuit(circuit, controllers=controllers)
    return CompiledCircuit(circuit, task=task, options=options, ideal_state=ideal_state,
                           recovery=recovery, controllers=controllers, sampling=sampling)


def _ordinary(value):
    if value is None: return None
    try: return math.ldexp(value[0], value[1])
    except OverflowError: return None


def _format_weight_summary(result, schedule, history_sampler):
    encoded = result['mean_scientific']; mean = _ordinary(encoded)
    if mean == 0 and encoded[0] != 0: mean = None
    result.update(mean=mean if result['completed'] else None,
        mean_scientific=dict(mantissa=encoded[0], exponent2=encoded[1]) if result['completed'] else None,
        convention='target/proposal original-noise likelihood ratio; exact_marginal histories from their true conditionals',
        likelihood_ratio=1 if schedule.sampling['method'] == 'physical' else None,
        ess_scope='noise weights only; not a guarantee of error-rate precision')
    if history_sampler == 'affine_support':
        result.update(convention='original-noise likelihood ratio times complete history mass / history proposal probability',
                      likelihood_ratio=None, ess_scope='complete importance weights; not a guarantee of target-fidelity precision')
    return result


def _print_sampling_progress(completed, shots, method, efficiency):
    kappa = efficiency['kappa']
    label = f'{kappa:.6g}' if kappa is not None else f"unavailable ({efficiency['status']})"
    print(f'[{completed}/{shots}] history_sampler={method} kappa={label}', file=sys.stderr, flush=True)
    print('High kappa means affine_support needs more draws than exact_marginal for the same precision; '
          'consider --history-sampler exact_marginal.', file=sys.stderr, flush=True)


# Counters a compiled circuit accumulates across calls; run() reports its own share.
_RUN_COUNTERS = ('queries', 'compilations', 'kernel_seconds', 'compile_seconds', 'decoder_seconds', 'support_seconds')


def run(circuit, *, shots=1, seed=0, task=None, options=None, controllers=None,
        ideal_state=None, recovery=None, sampling=None, progress_every=None, workers=1):
    """Draw shots histories and return the aggregate result dictionary.

    circuit is a path, a loaded CircuitFile or a CompiledCircuit.
    A compiled circuit already fixes its task, options, controllers, targets
    and sampling, so those arguments must then be omitted; repeated calls
    reuse its preparation. workers selects the number of native sampling
    threads; one preserves the sequential random stream.
    """
    from . import _native
    if type(shots) is not int or shots < 1: raise ValueError(f'shots must be a positive integer, not {shots!r}')
    if type(seed) is not int: raise ValueError(f'seed must be an integer, not {seed!r}')
    if type(workers) is not int or workers < 1:
        raise ValueError(f'workers must be a positive integer, not {workers!r}')
    if progress_every is not None and (type(progress_every) is not int or progress_every < 1):
        raise ValueError(f'progress_every must be a positive integer, not {progress_every!r}')
    started = monotonic(); failures = []; preparation_failed = False; baseline = None
    if isinstance(circuit, CompiledCircuit):
        fixed = [name for name, value in (('task', task), ('options', options), ('controllers', controllers),
                 ('ideal_state', ideal_state), ('recovery', recovery), ('sampling', sampling)) if value is not None]
        if fixed:
            raise ValueError('a compiled circuit already fixes its '+', '.join(fixed)+'; pass them to compile_circuit')
        compiled = circuit
        if compiled._compilation['status'] in ('validated', 'failed'):
            raise ValueError('this compiled circuit has not completed preparation; compile it again')
        baseline = compiled._execution.state(False)
        compiled._execution.begin_run()
    else:
        try:
            compiled = compile_circuit(circuit, task=task, options=options, controllers=controllers,
                                       ideal_state=ideal_state, recovery=recovery, sampling=sampling)
        except (ResourceLimit, MemoryError) as exc:
            if not hasattr(exc, 'compiled_task'): raise
            compiled = exc.compiled_task; preparation_failed = True
            failures.append(dict(phase='preparation', shot=None, type=type(exc).__name__, message=str(exc)))
    rng = Random(seed); batch = _native.make_file_batch()
    sampling_started = monotonic(); sampling_seconds = 0.; completed = 0
    compiled._configure_execution()
    try:
        while not preparation_failed and completed < shots:
            count = min(4096, shots-completed)
            if progress_every is not None:
                count = min(count, progress_every-completed % progress_every)
            try:
                completed = _native.advance_file_batch(compiled._execution.handle, batch, rng, count,
                                                      min(workers, 4096))
            except (RuntimeError, ValueError, ArithmeticError, MemoryError) as exc:
                completed = _native.file_batch_efficiency(batch)['completed']
                failures.append(dict(phase='sampling', shot=completed, type=type(exc).__name__, message=str(exc))); break
            if progress_every is not None and (completed % progress_every == 0 or completed == shots):
                sampling_seconds += monotonic()-sampling_started
                _print_sampling_progress(completed, shots, compiled.options['history_sampler'],
                                         _native.file_batch_efficiency(batch))
                sampling_started = monotonic()
    finally:
        state = compiled._sync_execution()
    used = {name: state[name]-(baseline[name] if baseline else 0) for name in _RUN_COUNTERS}
    sampling_seconds += monotonic()-sampling_started
    summary = _native.file_batch_summary(batch)
    completed = summary.pop('completed')
    weights = _format_weight_summary(summary.pop('weights'), compiled.noise_schedule, compiled.options['history_sampler'])
    efficiency = summary.pop('efficiency'); records = summary.pop('sampled_records')
    result = dict(schema_version=SCHEMA_VERSION, software_version=PACKAGE_VERSION,
        status='failed' if failures else 'complete', requested=shots, completed=completed,
        failed=0 if preparation_failed else len(failures),
        not_attempted=shots if preparation_failed else shots-completed-len(failures), seed=seed,
        workers=workers, route=compiled.route,
        method='global_boundary_with_noise_importance_sampling' if compiled.noise_schedule.sampling['method'] == 'importance'
               else 'global_boundary_with_independent_original_noise',
        sampling=compiled.noise_schedule.to_dict(history_sampler=compiled.options['history_sampler']),
        evaluation=compiled.evaluation(),
        normalization='mean joint contributions; conditional error is ratio of aggregate means',
        uncertainty='IID draw standard errors; conditional ratio uses paired delta method; null for fewer than two draws',
        weights=weights,
        history_sampler=compiled.options.get('history_sampler', 'affine_support'),
        efficiency=efficiency,
        fingerprints=dict(circuit=fingerprint(compiled._source_circuit.to_text()), task=fingerprint(compiled.task['raw']),
                          controllers=fingerprint({name:b.fingerprint for name,b in compiled.circuit.controllers.bindings.items()}), options=fingerprint(compiled.options),
                          ideal_state=None if compiled.ideal_state is None or compiled._error_observables is not None
                                      else fingerprint(compiled.ideal_state.to_dict()),
                          recovery=None if compiled.recovery is None else compiled.recovery.circuit.fingerprint),
        timings=dict(total_seconds=monotonic()-started, compile_seconds=used['compile_seconds'],
                     preparation_seconds=compiled._compilation.get('preparation_seconds'),
                     sampling_seconds=sampling_seconds,
                     query_seconds=used['kernel_seconds'], decoder_seconds=used['decoder_seconds'],
                     support_seconds=used['support_seconds']),
        resources=dict(queries=used['queries'], compilations=used['compilations'], limits=compiled.options,
                       compilation=deepcopy(compiled._compilation)), failures=failures,
        sampled_records=[list(row) for row in records] if compiled.task['sample_records'] else None,
        **summary)
    if failures:
        fields = ('estimates','standard_errors','estimates_scientific','standard_errors_scientific')
        result['partial_statistics'] = {k: result[k] for k in fields if k in result}
        for key in fields:
            if key in result: result[key] = None
    result['fingerprints']['sampling'] = fingerprint(compiled.noise_schedule.sampling)
    return result


def _number(value):
    return '—' if value is None else f'{value:.6g}'


def _seconds(value):
    return '—' if value is None else f'{value:.3g} s'


def _estimate_lines(values, errors, names):
    width = max(len(label) for _, label in names)
    lines = []
    for key, label in names:
        value = None if values is None else values.get(key)
        error = None if errors is None else errors.get(key)
        lines.append(f'  {label:<{width}}  {_number(value)}' + ('' if error is None else f' ± {_number(error)}'))
    return lines


def _failure_lines(result):
    return [f"  failure: {f['type']}: {f['message']}" for f in result.get('failures', ())[:3]]


def _summarize_run(result):
    lines = [f"run: {result['status']}, {result['completed']} of {result['requested']} draws, seed {result['seed']}"]
    estimates = result.get('estimates') or result.get('partial_statistics', {}).get('estimates')
    errors = result.get('standard_errors') or result.get('partial_statistics', {}).get('standard_errors')
    if result.get('partial_statistics'): lines.append('  partial statistics of the completed draws:')
    lines += _estimate_lines(estimates, errors, (('acceptance', 'acceptance'), ('error', 'error'),
                                                 ('conditional_error', 'conditional error')))
    if estimates and estimates.get('error') is None and result['status'] == 'complete':
        lines.append('  (no error is defined: the task has no ideal_state or error_observables)')
    if estimates and estimates.get('moments'):
        lines.append('  moments: '+', '.join(_number(v) for v in estimates['moments']))
    compilation = result['resources']['compilation']
    details = [f"route {result['route']}"]
    if compilation.get('sampled_history'):
        details.append(f"history sampler {result['history_sampler']}")
        if result['efficiency'].get('kappa') is not None: details.append(f"kappa {_number(result['efficiency']['kappa'])}")
    if result['sampling']['method'] == 'importance':
        details.append(f"importance sampling, proposal_scale {result['sampling']['proposal_scale']}")
    lines.append('  '+', '.join(details))
    timings = result['timings']
    lines.append(f"  time {_seconds(timings['total_seconds'])} (preparation {_seconds(timings['preparation_seconds'])}, "
                 f"sampling {_seconds(timings['sampling_seconds'])})")
    return lines + _failure_lines(result)


def _summarize_enumeration(result):
    lines = [f"enumerate: {result['status']}, faults up to weight {result['max_weight']}, "
             f"{result['completed_patterns']} patterns"]
    totals = result.get('totals') or result.get('partial_totals')
    if result['status'] != 'complete' and totals: lines.append('  partial totals of the completed patterns:')
    lines += _estimate_lines(totals, None, (('covered_probability', 'covered probability'), ('acceptance', 'acceptance'),
        ('error', 'error'), ('conditional_error_within_enumerated_sector', 'conditional error (enumerated)')))
    bounds = (totals or {}).get('full_distribution_bounds') or {}
    for name, value in bounds.items():
        if value is not None:
            lines.append(f"  {name.replace('_', ' ')} bounds over the full distribution: [{_number(value[0])}, {_number(value[1])}]")
    lines.append(f"  route {result['route']}, time {_seconds(result['timings']['total_seconds'])}")
    return lines + _failure_lines(result)


def _summarize_circuit(result, path):
    analysis = result['control_analysis']
    external = sorted({i['controller'] for i in analysis['external_inputs']})
    def count(n, noun): return f'{n} {noun}' + ('' if n == 1 else 's')
    return [f"{path}: {result['format']}, {count(result['num_qubits'], 'qubit')}, "
            f"{count(result['num_records'], 'measurement record')}, {count(len(analysis['detectors']), 'detector')}, "
            f"{count(len(result['instructions']), 'instruction')}",
            f"  route {result['route']}; decoders: {', '.join(external) if external else 'none'}"]


def _summarize_inspect(result, path):
    lines = [f'inspect: {result["compilation"]["status"]}'] + _summarize_circuit(result, path)
    evaluation = result['evaluation']
    lines.append(f"  outputs {evaluation['output_mapping']}; error metric: {evaluation['error_metric']}")
    compilation = result['compilation']
    if compilation.get('resource_summary'):
        resource = compilation['resource_summary']
        lines.append(f"  prepared queries {resource['queries']}, at most {resource['max_reference_terms']} reference terms "
                     f"and {resource['max_reference_phase_coordinates']} phase coordinates per query")
        lines.append(f"  preparation {_seconds(compilation['preparation_seconds'])}; "
                     f"history sampling {'needed' if compilation['sampled_history'] else 'not needed'}")
    return lines


def _summarize_comparison(result):
    lines = [f"compare-samplers: {result['status']}, {result['pilot_shots_per_channel']} pilot draws per sampler"]
    if result['status'] == 'complete':
        lines.append(f"  kappa {_number(result.get('kappa'))}, Gamma {_number(result.get('gamma'))}; "
                     f"recommended history sampler: {result['recommended_sampler']}")
    else:
        lines.append(f"  no recommendation: {result.get('reason', 'see the JSON result')}")
    for method, pilot in result['pilots'].items():
        lines.append(f"  {method}: sampling {_seconds(pilot['timings']['sampling_seconds'])}")
    return lines


def _parser():
    parser = argparse.ArgumentParser(prog='magna',
        description='Simulate noisy circuits with non-Clifford gates, measurements, Pauli noise and classical feedback.',
        epilog='Run "magna COMMAND --help" for the options of one command.')
    parser.add_argument('--version', action='version', version=f'magna {PACKAGE_VERSION}')
    commands = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')

    def command(name, summary, *, sampling=True, history_sampler=True):
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.add_argument('circuit', metavar='CIRCUIT', help='circuit file (.magna)')
        sub.add_argument('--task', metavar='FILE', help='task JSON: output qubits and error definition')
        sub.add_argument('--controllers', metavar='FILE', help='decoder bindings JSON for external CONTROL[name] functions')
        sub.add_argument('--ideal-state', metavar='FILE', help='noiseless circuit preparing the ideal target, if not named in the task')
        sub.add_argument('--recovery', metavar='FILE', help='noiseless terminal recovery circuit, if not named in the task')
        sub.add_argument('--options', metavar='FILE', help='resource-budget JSON')
        if sampling:
            sub.add_argument('--sampling', metavar='FILE',
                             help='noise proposal JSON, e.g. {"method": "importance", "proposal_scale": "10"}')
        if history_sampler:
            sub.add_argument('--history-sampler', choices=HISTORY_SAMPLERS,
                             help='how measurement histories are drawn when decoders need them (default affine_support)')
        output = sub.add_argument_group('output')
        output.add_argument('--output', metavar='FILE', help='also write the full JSON result to FILE')
        output.add_argument('--json', action='store_true', help='print the full JSON result instead of a summary')
        return sub

    command('validate', 'Parse the circuit and check the task, options and decoder bindings without compiling.')
    command('inspect', 'Compile the task and report the route and the prepared resources.')
    run_command = command('run', 'Sample the task and report acceptance, error and conditional error.')
    run_command.add_argument('--shots', type=int, default=1, metavar='N', help='number of draws (default 1)')
    run_command.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default 0)')
    run_command.add_argument('--workers', type=int, default=1, metavar='N',
                             help='native sampling threads (positive integer, default 1)')
    run_command.add_argument('--progress-every', type=int, metavar='N',
                             help='print progress and kappa to standard error every N draws')
    enumerate_command = command('enumerate', 'Sum every fault pattern up to a maximum weight exactly.',
                                sampling=False, history_sampler=False)
    enumerate_command.add_argument('--max-weight', type=int, required=True, metavar='K',
                                   help='maximum number of faults per pattern')
    enumerate_command.add_argument('--max-patterns', type=int, default=100000, metavar='N',
                                   help='pattern budget (default 100000)')
    enumerate_command.add_argument('--max-branches', type=int, default=4096, metavar='N',
                                   help='decoder-history budget (default 4096)')
    enumerate_command.add_argument('--include-patterns', action='store_true', help='list every pattern in the JSON result')
    compare = command('compare-samplers', 'Run a short pilot of both history samplers and recommend one.',
                      history_sampler=False)
    compare.add_argument('--pilot-shots', type=int, required=True, metavar='N', help='draws per history sampler (at least 2)')
    compare.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default 0)')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        controllers = load_controllers(args.controllers) if args.controllers else None
        circuit = load_circuit(args.circuit, controllers=controllers)
        options = _read_json(args.options)
        if getattr(args, 'history_sampler', None) is not None: options['history_sampler'] = args.history_sampler
        targets = dict(task=args.task, ideal_state=args.ideal_state, recovery=args.recovery)
        sampling = getattr(args, 'sampling', None)
        if args.command == 'enumerate':
            from .low_weight import enumerate_errors
            result = enumerate_errors(circuit, max_weight=args.max_weight, options=options, max_patterns=args.max_patterns,
                                      max_branches=args.max_branches, include_patterns=args.include_patterns, **targets)
            lines = _summarize_enumeration(result)
        elif args.command == 'run':
            result = run(circuit, shots=args.shots, seed=args.seed, options=options, sampling=sampling,
                         progress_every=args.progress_every, workers=args.workers, **targets)
            lines = _summarize_run(result)
        elif args.command == 'compare-samplers':
            from .sampler_decision import compare_samplers
            result = compare_samplers(circuit, pilot_shots=args.pilot_shots, seed=args.seed,
                                      options=options, sampling=sampling, **targets)
            lines = _summarize_comparison(result)
        elif args.command == 'inspect':
            result = compile_circuit(circuit, options=options, sampling=sampling, **targets).inspect()
            lines = _summarize_inspect(result, args.circuit)
        else:
            compiled = CompiledCircuit(circuit, options=options, sampling=sampling, _prepare=False, **targets)
            result = dict(schema_version=SCHEMA_VERSION, status='valid', **compiled.inspect())
            lines = ['valid: '+_summarize_circuit(result, args.circuit)[0]] + _summarize_circuit(result, args.circuit)[1:]
        content = json.dumps(result, indent=2, allow_nan=False, default=str)+'\n'
        if args.output:
            Path(args.output).write_text(content)
            lines.append(f'Full result written to {args.output}.')
        if args.json: print(content, end='')
        else: print('\n'.join(lines))
        return 1 if result.get('status') == 'failed' else 0
    except (ResourceLimit, MemoryError, ArithmeticError, RuntimeError) as exc:
        # A budget, memory, numerical or internal failure on valid input: the command ran but could not finish.
        if args.json: print(json.dumps(dict(schema_version=1, status='failed', error=f'{type(exc).__name__}: {exc}')))
        else: print(f'magna {args.command}: failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        if args.json: print(json.dumps(dict(schema_version=1, status='invalid', error=str(exc))))
        else: print(f'magna {args.command}: error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
