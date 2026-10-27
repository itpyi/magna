"""No-feedback packing must preserve actual records, ownership and budgets."""
from fractions import Fraction
from unittest.mock import patch
import unittest
import weakref

from magna.operations import (InstrumentIR, StabilizerKet, Simulator, Options, SampleAssignment,
                                        NoiseModel, NoiseSite)
from magna.batch import CompactAssignment
try:
    from magna import _native
except ImportError:
    _native = None


@unittest.skipUnless(_native is not None and hasattr(_native, 'make_full_record'),
                     'native global extension required')
class NativeGlobalFastpathTests(unittest.TestCase):
    @staticmethod
    def instrument(exponent=1):
        return InstrumentIR((0,), (0,), (
            dict(id='t', kind='T', qubits=[0], exponent=exponent),
            dict(id='m', kind='MX', qubits=[0], record='m')),
            StabilizerKet.positive(1, (1,)))

    def simulator(self):
        sim = Simulator(self.instrument(), Options(cache_entries=0))
        sim.prepare()
        return sim

    def test_shared_immutable_faults_keep_distinct_records_and_batch_scope(self):
        noise = NoiseModel((NoiseSite('t', (0,), (('Z',),), Fraction(1, 10)),))
        reference = Simulator(self.instrument(), Options(reuse_structure=False))
        sim = self.simulator(); plan = sim._prepared.native_plan
        assignments = list(sim.compact_noise_assignments(noise, [[(0, 1)]], (0, 1)))
        self.assertIs(assignments[0].faults, assignments[1].faults)
        expected = [reference.probability(a) for a in assignments]
        self.assertNotEqual(*expected)
        before = plan.info['packed_fault_groups']
        found, _, count = plan.evaluate(assignments)
        self.assertEqual(found, expected); self.assertEqual(count, 2)
        self.assertEqual(plan.info['packed_fault_groups']-before, 1)
        # The next batch packs again: nothing is cached on the plan.
        self.assertEqual(plan.evaluate(assignments)[0], expected)
        self.assertEqual(plan.info['packed_fault_groups']-before, 2)

    def test_mutable_fault_tuple_is_repacked_after_each_yield(self):
        sim = self.simulator(); plan = sim._prepared.native_plan
        reference = Simulator(self.instrument(), Options(reuse_structure=False))
        expected = [reference.probability(SampleAssignment({'m': 0},
            (dict(after='t', qubit=0, pauli=p),))) for p in ('I', 'Z')]
        self.assertNotEqual(*expected)
        fault = dict(after='t', qubit=0, pauli='I')
        assignment = SampleAssignment({'m': 0}, (fault,))
        fault = assignment.faults[0]
        def inputs():
            fault['pauli'] = 'I'; yield assignment
            fault['pauli'] = 'Z'; yield assignment
        before = plan.info['packed_fault_groups']
        self.assertEqual(plan.evaluate(inputs())[0], expected)
        self.assertEqual(plan.info['packed_fault_groups']-before, 2)
        self.assertEqual(list(sim.probabilities(inputs())), expected)

    def test_snapshot_check_precedes_shared_empty_fault_reuse(self):
        sim = self.simulator()
        foreign = Simulator(self.instrument(3), Options(cache_entries=0))._prepare()
        first = CompactAssignment(sim._prepared, 0)
        second = CompactAssignment(foreign, 0)
        # Establish shared storage explicitly; empty-tuple interning varies by Python version.
        object.__setattr__(second, 'faults', first.faults)
        self.assertIs(first.faults, second.faults)
        with self.assertRaisesRegex(ValueError, 'different IR snapshot'):
            sim._prepared.native_plan.evaluate([first, second])

    def test_original_wire_lifetime_is_still_checked_by_native_kernel(self):
        sim = self.simulator()
        invalid = CompactAssignment(sim._prepared, 0,
            (dict(after='t', qubit=9, pauli='X'),))
        with self.assertRaises(ValueError):
            sim.probability(invalid)

    def test_ordinary_record_validation_and_empty_stream(self):
        sim = self.simulator(); plan = sim._prepared.native_plan
        for records in ({}, {'m': 2}, {'m': True}, {'m': 0, 'extra': 0}):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    sim.probability(SampleAssignment(records))
        with self.assertRaises(ValueError):
            sim.probability(SampleAssignment({'m': 0}, report_flips=('unknown',)))
        with self.assertRaises(ValueError):
            sim.probability(SampleAssignment({'m': 0},
                (dict(after='unknown', qubit=0, pauli='Z'),)))
        values, timings, count = plan.evaluate(iter(()))
        self.assertEqual(values, []); self.assertEqual(count, 0)
        self.assertEqual(timings['feedback_fallback_seconds'], 0)

    def test_no_feedback_does_not_retain_consumed_assignments(self):
        sim = self.simulator(); plan = sim._prepared.native_plan
        refs = []
        def inputs():
            for i in range(64):
                assignment = SampleAssignment({'m': i % 2})
                refs.append(weakref.ref(assignment))
                yield assignment
                self.assertLessEqual(sum(ref() is not None for ref in refs), 1)
        values, _, count = plan.evaluate(inputs())
        self.assertEqual(len(values), 64); self.assertEqual(count, 64)
        self.assertTrue(all(ref() is None for ref in refs))


if __name__ == '__main__':
    unittest.main()
