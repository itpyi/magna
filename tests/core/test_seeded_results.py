"""The seeded comparison tells results apart from counters and fingerprints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.seeded_results import (CASES, ROOT, category, compare, describe, differences, prepare, render, review,
                                    says_unchanged, short)

UNCHANGED = '- Results of a seeded run: unchanged\n'
CHANGED = '- Results of a seeded run: changed. Draws now use a new stream.\n'


def fake_run(old, new):
    """Stand in for run_case: the base checkout returns one result, the head checkout another."""
    def run(tree, folder, options, shots, output):
        return old if 'main' in output.name else new
    return mock.patch('scripts.seeded_results.run_case', run)


class SeededResultsTests(unittest.TestCase):
    def test_every_case_exists(self):
        for name, folder, _ in CASES:
            self.assertTrue((ROOT / folder).is_dir(), name)

    def test_timings_and_the_version_are_ignored(self):
        old = {'timings': {'total': 1}, 'compile_seconds': 1, 'software_version': '0.2.0', 'error': 0.5}
        new = {'timings': {'total': 2}, 'compile_seconds': 2, 'software_version': '0.3.0', 'error': 0.5}
        self.assertEqual(list(differences(old, new)), [])

    def test_differences_name_nested_paths(self):
        old = {'a': [{'b': 1}, {'b': 2}], 'c': 1}
        new = {'a': [{'b': 1}, {'b': 3}], 'd': 1}
        self.assertEqual(list(differences(old, new)), [('a[1].b', 2, 3), ('c', 1, None), ('d', None, 1)])

    def test_lists_of_different_lengths_are_one_difference(self):
        self.assertEqual(list(differences({'a': [1]}, {'a': [1, 2]})), [('a', [1], [1, 2])])

    def test_categories(self):
        self.assertEqual(category('estimates.error'), 'Results')
        self.assertEqual(category('resources.queries'), 'Counters')
        self.assertEqual(category('fingerprints.ideal_state'), 'Fingerprints')
        self.assertEqual(category('evaluation.recovery.fingerprint'), 'Fingerprints')

    def test_compare_sorts_changes(self):
        found = compare({'estimates': {'error': 1}, 'resources': {'queries': 2}},
                        {'estimates': {'error': 1}, 'resources': {'queries': 3}})
        self.assertEqual(found, {'Results': [], 'Counters': [('resources.queries', 2, 3)], 'Fingerprints': []})

    def test_description_answers(self):
        self.assertTrue(says_unchanged(UNCHANGED))
        self.assertTrue(says_unchanged('- Seeded numerical results, RNG state and decoder calls: unchanged; fewer queries.'))
        self.assertFalse(says_unchanged('- Results of a seeded run: changed only in `fingerprints.ideal_state`'))
        self.assertFalse(says_unchanged(None))

    def test_fixture_gets_a_run_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'task'
            prepare(ROOT / 'tests/fixtures/importance-sampling', target)
            self.assertEqual(json.loads((target / 'run.json').read_text())['sampling']['method'], 'importance')

    def test_changed_result_contradicts_an_unchanged_answer(self):
        with fake_run(({'estimates': {'error': 0.1}}, None), ({'estimates': {'error': 0.2}}, None)):
            rows, details, _, problems = review(ROOT, ROOT, UNCHANGED, 8)
        self.assertEqual(rows[0][1:], ('1 changed', 'same', 'same'))
        self.assertIn('`estimates.error` 0.1 → 0.2', details[0])
        self.assertEqual(len(problems), 1)
        self.assertIn('dk', problems[0])

    def test_declared_change_is_listed_without_a_problem(self):
        with fake_run(({'estimates': {'error': 0.1}}, None), ({'estimates': {'error': 0.2}}, None)):
            rows, details, _, problems = review(ROOT, ROOT, CHANGED, 8)
        self.assertEqual(len(details), len(CASES))
        self.assertEqual(problems, [])

    def test_counter_change_is_no_contradiction(self):
        with fake_run(({'resources': {'queries': 2}}, None), ({'resources': {'queries': 1}}, None)):
            rows, _, _, problems = review(ROOT, ROOT, UNCHANGED, 8)
        self.assertEqual(rows[0][1:], ('same', '1 changed', 'same'))
        self.assertEqual(problems, [])

    def test_long_lists_of_changes_are_shortened(self):
        old = {'estimates': {f'value{i}': 0 for i in range(8)}}
        new = {'estimates': {f'value{i}': 1 for i in range(8)}}
        with fake_run((old, None), (new, None)):
            _, details, complete, _ = review(ROOT, ROOT, CHANGED, 8)
        self.assertTrue(details[3].endswith('5 more'))
        self.assertEqual(len(complete), 8 * len(CASES))
        self.assertIn('`estimates.value7` 0 → 1', complete[7])

    def test_estimates_are_listed_first(self):
        found = compare({'efficiency': {'kappa': 1}, 'estimates': {'error': 1}}, {'efficiency': {'kappa': 2}, 'estimates': {'error': 2}})
        self.assertIn('estimates.error', describe('dk', found)[0][0])

    def test_values_keep_full_precision_but_not_whole_fingerprints(self):
        self.assertEqual(short(0.8805541992187513), '0.8805541992187513')
        self.assertEqual(short('a' * 64), 'a' * 24 + '…')

    def test_task_that_cannot_run_is_reported(self):
        with fake_run(({}, None), (None, 'ValueError: broken')):
            rows, _, _, problems = review(ROOT, ROOT, UNCHANGED, 8)
        self.assertEqual(rows[0][1:], ('did not run',) * 3)
        self.assertIn('did not run on this pull request: ValueError: broken', problems[0])

    def test_render_says_when_nothing_is_open(self):
        text = render([('dk', 'same', 'same', 'same')], [], [], [], 8)
        self.assertIn('Nothing open.', text)
        self.assertNotIn('### Differences', text)
        self.assertNotIn('<details>', text)

    def test_render_keeps_every_difference_with_full_values(self):
        value = 'a' * 64
        found = compare({'fingerprints': {'task': value}}, {'fingerprints': {'task': 'b'}})
        text = render([('dk', 'same', 'same', '1 changed')], *describe('dk', found), [], 8)
        self.assertIn('<details><summary>Every difference</summary>', text)
        self.assertIn(f'{value} → b', text)

    def test_checkout_matches_itself(self):
        rows, details, complete, problems = review(ROOT, ROOT, UNCHANGED, 64)
        self.assertEqual({row[1:] for row in rows}, {('same', 'same', 'same')})
        self.assertEqual((details, complete, problems), ([], [], []))


if __name__ == '__main__':
    unittest.main()
