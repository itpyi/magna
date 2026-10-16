"""File syntax, physical site identity and static control information tests."""
from fractions import Fraction
import json
import tempfile
import unittest
from pathlib import Path

from magna.circuit_file import (
    AffineExpr, CircuitSyntaxError, ControllerBinding, ControllerRegistry,
    Target, load_circuit, parse_circuit,
)


class CircuitFileTests(unittest.TestCase):
    def test_native_rationals_and_replaced_views(self):
        from dataclasses import replace
        for literal in ('1e-200', '.125', '+0.1250', '1_000/100_000',
                        '123456789012345678901234567890/123456789012345678901234567891'):
            circuit = parse_circuit(f'X_ERROR({literal}) 0')
            # Fraction only accepts underscores from Python 3.11 onward.
            self.assertEqual(circuit.instructions[0].args, (Fraction(literal.replace('_', '')),))
        circuit = parse_circuit('M 0\nCONTROL[NOT] rec[-1] -> ctrl[0]')
        rewritten = replace(circuit, instructions=circuit.instructions[:1])
        with self.assertRaisesRegex(ValueError, 'no parsed form'):
            rewritten.format_version
        analysis = replace(circuit.analysis, affine={Target('ctrl', 0, 0): AffineExpr()})
        self.assertEqual(analysis.expression(Target('ctrl', 0, 0)), AffineExpr())

    def test_lossless_roundtrip_and_resolved_repeat(self):
        text = "# physical sites\nR 0 1\nM 0\nCONTROL[XOR] rec[-1] -> ctrl[7]\nREPEAT 2 {\n MX(0.1) 1\n CONTROL[XOR](1) ctrl[7] rec[-1] -> ctrl[7]\n CZ ctrl[7] 0\n}\n"
        circuit = parse_circuit(text, source="example.magna")
        self.assertEqual(circuit.to_text(), text)
        self.assertEqual((circuit.num_records, circuit.num_qubits), (3, 2))
        controls = [i for i in circuit.instructions if i.name == "CONTROL"]
        self.assertEqual([i.outputs[0].version for i in controls], [0, 1, 2])
        self.assertEqual(controls[2].targets[0], controls[1].outputs[0])
        self.assertEqual(controls[1].source.repeat, (0,))
        self.assertEqual(controls[2].source.repeat, (1,))
        self.assertEqual(controls[1].source.line, 7)
        expr = circuit.analysis.expression(controls[-1].outputs[0])
        self.assertEqual(expr, AffineExpr(0, frozenset(Target("rec", k) for k in range(3))))
        self.assertEqual(len({i.op_id for i in circuit.instructions}), len(circuit.instructions))
        json.dumps(circuit.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.magna"
            circuit.write(path)
            self.assertEqual(load_circuit(path).fingerprint, circuit.fingerprint)

    def test_atomic_outputs_and_nonlinear_parity_information(self):
        calls = []
        registry = ControllerRegistry()
        registry.register("actual", lambda bits, config: calls.append(bits) or bits,
                          version="1", config={"algorithm": "specified"})
        circuit = parse_circuit("M 0 1 2\nCONTROL[XOR] rec[-3] rec[-2] -> ctrl[0]\n"
                                "DETECTOR rec[-1]\n"
                                "CONTROL[actual] ctrl[0] det[-1] -> ctrl[0] ctrl[1]\n"
                                "CONTROL[XOR] ctrl[0] ctrl[1] -> ctrl[2]\n"
                                "CH ctrl[2] 0\n", controllers=registry)
        self.assertEqual(calls, [])
        self.assertEqual(circuit.analysis.route, "nonlinear")
        self.assertEqual(len(circuit.analysis.required_parities), 2)
        self.assertEqual(circuit.analysis.required_parities[0].terms,
                         frozenset((Target("rec", 0), Target("rec", 1))))
        external = circuit.analysis.external_instructions[0]
        self.assertEqual(external.targets[0].version, 0)
        self.assertEqual(external.outputs[0].version, 1)
        expr = circuit.analysis.expression(circuit.instructions[-1].targets[0])
        self.assertEqual(expr.terms, frozenset(external.outputs))
        self.assertEqual(circuit.controllers.resolve("actual").invoke((0, 1), 2), (0, 1))
        self.assertEqual(calls, [(0, 1)])

    def test_xor_cancellation_constants_and_inversion(self):
        circuit = parse_circuit("M 0\nCONTROL[XOR](1) rec[-1] rec[-1] -> ctrl[0]\n"
                                "CONTROL[XOR] !ctrl[0] -> ctrl[1]\nCX ctrl[1] 0\n")
        self.assertEqual(circuit.analysis.route, "linear")
        self.assertEqual(circuit.analysis.expression(circuit.instructions[-1].targets[0]), AffineExpr())
        self.assertEqual(circuit.num_records, 1)

    def test_measurement_record_counts_products_and_exact_probability(self):
        circuit = parse_circuit("R 0 1 2\nMXX(0.01) 0 1\nMPP(1/30) !X0*X1 Z2\n"
                                "MRX 0 1\nMY !2\nDETECTOR rec[-1] rec[-4]\n"
                                "OBSERVABLE_INCLUDE(4) rec[-2]\n")
        self.assertEqual(circuit.num_records, 6)
        self.assertEqual(circuit.instructions[1].args, (Fraction(1, 100),))
        mpp = circuit.instructions[2]
        self.assertEqual(mpp.records, (1, 2))
        self.assertEqual(mpp.args, (Fraction(1, 30),))
        self.assertTrue(mpp.products[0][0].inverted)
        self.assertEqual(circuit.instructions[-2].targets, (Target("rec", 5), Target("rec", 2)))

    def test_builtin_not_is_affine_versioned_and_requires_no_binding(self):
        text = ('M 0\nDETECTOR rec[-1]\nCONTROL[NOT] det[0] -> ctrl[0]\n'
                'REPEAT 2 {\nCONTROL[NOT] ctrl[0] -> ctrl[0]\n}\n'
                'CONTROL[NOT] !rec[-1] -> ctrl[1]\n')
        circuit = parse_circuit(text)
        self.assertEqual(circuit.format_version, 'magna-0.1')
        self.assertEqual(circuit.analysis.route, 'linear')
        self.assertFalse(circuit.analysis.external_instructions)
        writes = [i for i in circuit.instructions if i.name == 'CONTROL']
        self.assertEqual([i.outputs[0].version for i in writes], [0, 1, 2, 0])
        for inst, bit in zip(writes, (1, 0, 1, 0)):
            self.assertEqual(circuit.analysis.expression(inst.outputs[0]),
                             AffineExpr(bit, frozenset((Target('rec', 0),))))
        self.assertEqual(parse_circuit(text).fingerprint, circuit.fingerprint)
        json.dumps(circuit.to_dict())
        with self.assertRaisesRegex(ValueError, 'reserved'):
            ControllerRegistry().register('NOT', lambda bits, config: bits, version='1')

    def test_builtin_not_arity_and_future_inputs_are_rejected(self):
        for text in ('CONTROL[NOT] -> ctrl[0]',
                     'M 0\nCONTROL[NOT] rec[-1] rec[-1] -> ctrl[0]',
                     'M 0\nCONTROL[NOT] rec[-1] -> ctrl[0] ctrl[1]',
                     'M 0\nCONTROL[NOT](1) rec[-1] -> ctrl[0]',
                     'CONTROL[NOT] ctrl[0] -> ctrl[0]'):
            with self.subTest(text=text), self.assertRaises(CircuitSyntaxError):
                parse_circuit(text)

    def test_correlated_chain_and_conditional_channels(self):
        circuit = parse_circuit("M 0\nCONTROL[XOR] rec[-1] -> ctrl[0]\n"
                                "E(0.1) ctrl[0] X0 Y1\n"
                                "ELSE_CORRELATED_ERROR(2/9) ctrl[0] Z1\n"
                                "X_ERROR(0.001) 0\n")
        first, second, independent = circuit.instructions[2:]
        self.assertEqual(first.chain_id, second.chain_id)
        self.assertEqual(first.guard, Target("ctrl", 0, 0))
        self.assertEqual(first.targets, (Target("pauli", 0, pauli="X"), Target("pauli", 1, pauli="Y")))
        self.assertIsNone(independent.guard)
        self.assertEqual(first.args[0] + (1 - first.args[0]) * second.args[0], Fraction(3, 10))

    def test_bindings_are_versioned_and_config_isolated(self):
        config = {"a": [1]}
        def callback(bits, c):
            result = c["a"][0]
            c["a"][0] = 0
            return (result,)
        binding = ControllerBinding("decoder", "r1", callback, config)
        config["a"][0] = 0
        self.assertEqual(binding.invoke((), 1), (1,))
        self.assertEqual(binding.invoke((), 1), (1,))
        reg = ControllerRegistry([binding])
        a = parse_circuit("CONTROL[decoder] -> ctrl[0]", controllers=reg)
        b = parse_circuit("CONTROL[decoder] -> ctrl[0]", controllers=ControllerRegistry([
            ControllerBinding("decoder", "r2", callback, {"a": [1]})]))
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        with self.assertRaisesRegex(ValueError, "already registered"):
            reg.register("decoder", callback, version="r2")
        for callback in (lambda b, c: (2,), lambda b, c: (0.0,), lambda b, c: (), lambda b, c: None):
            binding = ControllerBinding("broken", "1", callback)
            with self.assertRaisesRegex(ValueError, "broken"):
                binding.invoke((), 1)

    def test_s_and_t_are_distinct_and_aliases_are_canonical(self):
        c = parse_circuit("S 0\nT 0\nS_DAG 0\nT_DAG 0\nCNOT 0 1\nCCX 0 1 2\n"
                          "CCZ 0 1 2\nCS_DAG 0 1\nCH 1 2\nMZ 0\n")
        self.assertEqual([i.name for i in c.instructions],
                         ["S", "T", "S_DAG", "T_DAG", "CX", "CCX", "CCZ", "CS_DAG", "CH", "M"])

    def test_stim_subset_matches_record_count(self):
        try:
            import stim
        except ImportError:
            self.skipTest("Stim is optional; install it for compatibility validation")
        text = "R 0 1 2\nH 0\nCNOT 0 1\nS 1\nMXX(0.01) 0 1\nMPP X0*X1 Z2\nREPEAT 3 {\nMR 0\nDETECTOR rec[-1] rec[-2]\n}\n"
        ours = parse_circuit(text)
        reference = stim.Circuit(text)
        self.assertEqual(ours.num_records, reference.num_measurements)
        self.assertEqual(ours.num_qubits, reference.num_qubits)

    def test_two_qubit_classical_controls_are_versioned_and_validate_targets(self):
        text = 'M 0\nCONTROL[XOR] rec[-1] -> ctrl[0]\nCCZ rec[-1] 1 2 !ctrl[0] 2 3\nCCX ctrl[0] 3 1\n'
        circuit = parse_circuit(text)
        self.assertEqual(circuit.format_version, 'magna-0.1')
        self.assertEqual(circuit.to_text(), text)
        self.assertEqual(circuit.num_qubits, 4)
        self.assertEqual(parse_circuit('CCZ 0 1 2\nCCX 0 1 2').format_version, 'magna-0.1')
        for name in ('CCZ', 'CCX'):
            for targets in ('rec[-1] 1 1', 'rec[-1] rec[-1] 1', '0 rec[-1] 1', 'rec[-1] !1 2'):
                with self.assertRaises(CircuitSyntaxError):
                    parse_circuit('M 0\n'+name+' '+targets)

    def test_source_located_rejections(self):
        invalid = {
            "CX rec[-1] 0": "earlier measurement",
            "CONTROL[XOR] ctrl[0] -> ctrl[0]": "uninitialized",
            "CONTROL[XOR] -> ctrl[0] ctrl[1]": "exactly one",
            "CONTROL[missing] -> ctrl[0]": "missing controller",
            "X_ERROR(0.1)": "requires targets",
            "PAULI_CHANNEL_1(0.6,0.3,0.2) 0": "sum",
            "DEPOLARIZE2(0.1) 0": "groups of 2",
            "M(1.1) 0": "report-flip",
            "S(0.1) 0": "does not take arguments",
            "CX 0 0": "repeats a qubit",
            "MPP X0*Y0": "repeats a qubit",
            "MPP 0": "Pauli targets",
            "X_ERROR(0.1) X0": "quantum target",
            "ELSE_CORRELATED_ERROR(0.1) X0": "immediately follow",
            "IF ctrl[0]": "unsupported instruction",
            "REPEAT 2 {\nH 0": "unclosed",
            "REPEAT 2\nH 0": "requires '{'",
            "}": "unmatched",
            "T !0": "inverted target",
        }
        for text, reason in invalid.items():
            with self.subTest(text=text), self.assertRaisesRegex(CircuitSyntaxError, reason) as caught:
                parse_circuit(text, source="bad.magna")
            self.assertIn("bad.magna:", str(caught.exception))
        with self.assertRaisesRegex(CircuitSyntaxError, "same control guard"):
            parse_circuit("M 0\nE(0.1) rec[-1] X0\nELSE_CORRELATED_ERROR(0.1) X0")

    def test_expansion_budget_is_enforced(self):
        from magna.file_api import ResourceLimit
        with self.assertRaises(ResourceLimit) as stop:
            parse_circuit("REPEAT 20 {\nH 0\n}", max_instructions=10)
        self.assertEqual((stop.exception.option, stop.exception.limit, stop.exception.needed), ("max_instructions", 10, 20))
        with self.assertRaisesRegex(ResourceLimit, "max_instructions = 10"):
            parse_circuit("REPEAT 4 {\nREPEAT 4 {\nH 0\n}\n}", max_instructions=10)
        with self.assertRaises(ResourceLimit) as stop:
            parse_circuit("H 0\nM 70000\n", source="sparse.magna")
        self.assertEqual((stop.exception.option, stop.exception.limit, stop.exception.needed), ("max_qubits", 65536, 70001))
        self.assertIn("sparse.magna:2: qubit index 70000", str(stop.exception))
        self.assertEqual(parse_circuit("M 70000\n", max_qubits=70001).num_qubits, 70001)
        self.assertEqual(parse_circuit("REPEAT 999999999999999999 {\nREPEAT 999999999999999999 {\n}\n}").instructions, ())

    def test_cz_allows_classical_endpoint_in_either_position(self):
        circuit = parse_circuit("M 0\nCZ 1 rec[-1]\n")
        self.assertEqual(circuit.instructions[-1].targets, (Target("rec", 0), Target("qubit", 1)))


if __name__ == "__main__":
    unittest.main()
