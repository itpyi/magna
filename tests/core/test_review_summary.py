"""The review summary reads the description shapes pull requests really use."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.review_summary import ROOT, TEMPLATE, compare, maintainer_notes, matches, read_parts, render, review

PARTS = read_parts(TEMPLATE.read_text())
RUNTIME = 'core/magna/native/runtime/plan.hpp'
CUDA = 'core/magna/native/cuda/sampler.cu'
ROUTINE = ['CHANGELOG.md', 'tests/core/test_plan.py']
BOXES = ('- [x] Kernel `core/magna/native/runtime/`\n- [ ] Bindings `core/magna/native/bindings/`\n'
         '- [x] Project documents `CHANGELOG.md`\n- [x] Tests `tests/`\n')


def description(scope=BOXES, interface='- Public interface: unchanged\n'):
    return (f'## Purpose\n\nFaster.\n\n## Scope\n\n{scope}\n'
            f'## Interface, format and results\n\n{interface}\n## Tests\n\nAdded one.\n')


def open_items(files, body):
    return review(files, body, PARTS)[1]


class ReviewSummaryTests(unittest.TestCase):
    def test_template_covers_every_tracked_file(self):
        files = subprocess.run(('git', 'ls-files'), cwd=ROOT, check=True, capture_output=True, text=True).stdout
        patterns = [pattern for _, part in PARTS for pattern in part]
        self.assertEqual([f for f in files.splitlines() if not any(matches(f, p) for p in patterns)], [])

    def test_path_patterns(self):
        self.assertTrue(matches('core/magna/native/runtime/a.hpp', 'core/magna/native/runtime/'))
        self.assertTrue(matches('core/magna/run.py', 'core/magna/*.py'))
        self.assertFalse(matches('core/magna/native/run.py', 'core/magna/*.py'))
        self.assertFalse(matches('tests/README.md', 'README.md'))

    def test_complete_description_leaves_nothing_open(self):
        self.assertEqual(open_items([RUNTIME, *ROUTINE], description()), [])

    def test_file_in_an_unticked_part_is_reported(self):
        rows, items = review([RUNTIME, CUDA, *ROUTINE], description(), PARTS)
        self.assertEqual(len(items), 1)
        self.assertIn('CUDA sampler', items[0])
        self.assertIn(('CUDA sampler', 1, 'no'), rows)

    def test_part_can_be_partly_declared(self):
        body = description(BOXES + '- [x] Python interface `core/magna/file_api.py`\n')
        rows, items = review([RUNTIME, 'core/magna/file_api.py', 'core/magna/other.py', *ROUTINE], body, PARTS)
        self.assertIn('1 of 2', [declared for _, _, declared in rows])
        self.assertEqual(len(items), 1)

    def test_outside_file_named_with_a_reason_is_declared(self):
        body = description(BOXES + '\nChanged files outside these parts:\n\n`core/magna/native/cuda/sampler.cu`: shared.\n')
        self.assertEqual(open_items([RUNTIME, CUDA, *ROUTINE], body), [])

    def test_comments_declare_nothing(self):
        body = description(BOXES + '<!-- Example: `core/magna/native/cuda/sampler.cu`: shared. -->\n')
        self.assertEqual(len(open_items([RUNTIME, CUDA, *ROUTINE], body)), 1)

    def test_description_saved_by_the_web_editor(self):
        self.assertEqual(open_items([RUNTIME, *ROUTINE], description().replace('\n', '\r\n')), [])

    def test_scope_written_as_prose_asks_for_boxes(self):
        rows, items = review([RUNTIME, *ROUTINE], description('Seven files: two runtime modules.\n'), PARTS)
        self.assertEqual(len(items), 1)
        self.assertIn('Tick the parts', items[0])
        self.assertEqual({declared for _, _, declared in rows}, {'?'})

    def test_box_without_a_path_is_not_guessed(self):
        items = open_items([RUNTIME, *ROUTINE], description('- [x] Kernel evaluation and runtime\n'))
        self.assertEqual(items, ['Name the path for this box under "Scope": Kernel evaluation and runtime'])

    def test_missing_section_is_reported(self):
        items = open_items(['README.md', 'CHANGELOG.md'], '## Purpose\n\nFix.\n')
        self.assertEqual(sum('section from the pull request template' in item for item in items), 3)

    def test_empty_description_is_reported(self):
        self.assertEqual(sum('section from the pull request template' in item for item in open_items(ROUTINE, None)), 4)

    def test_file_outside_every_part_is_shown(self):
        rows, items = review(['notes.txt', *ROUTINE], description(), PARTS)
        self.assertIn(('Outside every part', 1, 'no'), rows)
        self.assertEqual(len(items), 1)

    def test_unfilled_template_is_reported(self):
        items = open_items(ROUTINE, TEMPLATE.read_text())
        self.assertTrue(any('Tick the parts' in item for item in items))
        self.assertTrue(any('Choose one answer' in item for item in items))

    def test_changelog_needs_a_line_or_an_answer(self):
        files = [RUNTIME, 'tests/core/test_plan.py']
        self.assertTrue(any('CHANGELOG.md' in item for item in open_items(files, description())))
        answered = description(interface='- `CHANGELOG.md`: not needed\n')
        self.assertFalse(any('CHANGELOG.md' in item for item in open_items(files, answered)))

    def test_interface_change_needs_the_manual(self):
        body = description(interface='- Public interface (Python API, command line, result fields): changed. New option.\n')
        self.assertTrue(any('docs/software/' in item for item in open_items([RUNTIME, *ROUTINE], body)))
        scope = BOXES + '- [x] User manual `docs/`\n'
        body = body.replace(BOXES, scope)
        self.assertEqual(open_items([RUNTIME, 'docs/software/run.md', *ROUTINE], body), [])

    def test_format_change_needs_the_manual(self):
        changed = description(interface='- Public interface: unchanged\n- Circuit or task file format: changed\n')
        self.assertTrue(any('docs/software/' in item for item in open_items([RUNTIME, *ROUTINE], changed)))
        unchanged = description(interface='- Public interface: unchanged\n- Circuit and task formats: unchanged.\n')
        self.assertEqual(open_items([RUNTIME, *ROUTINE], unchanged), [])

    def test_package_change_needs_a_test(self):
        self.assertTrue(any('`tests/`' in item for item in open_items([RUNTIME, 'CHANGELOG.md'], description())))

    def test_kernel_change_needs_the_benchmark_suite(self):
        self.assertEqual(maintainer_notes([RUNTIME], 0, []),
                         ['This changes the C++ kernel, so it needs the benchmark suite before merging.'])

    def test_python_change_may_need_the_benchmark_suite(self):
        notes = maintainer_notes(['core/magna/boundary_sampling.py'], 0, [])
        self.assertEqual(len(notes), 1)
        self.assertIn('if that code runs during sampling', notes[0])

    def test_cuda_change_needs_gpu_measurements(self):
        for path in (CUDA, 'core/magna/cuda_sampler.py', 'core/magna/_cuda_noise.py'):
            notes = maintainer_notes([path], 0, [])
            self.assertEqual(len(notes), 1)
            self.assertIn('GPU measurements', notes[0])

    def test_kernel_and_cuda_change_gets_both_notes(self):
        self.assertEqual(len(maintainer_notes([RUNTIME, CUDA], 0, [])), 2)

    def test_tests_and_documents_need_no_note(self):
        self.assertEqual(maintainer_notes(ROUTINE, 0, []), [])

    def test_note_names_files_changed_on_both_sides(self):
        note = maintainer_notes(['README.md'], 2, ['README.md'])[0]
        self.assertIn('2 newer commits', note)
        self.assertIn('`README.md`', note)

    def test_render_says_when_nothing_is_open(self):
        self.assertIn('Nothing open.', render('1 file changed', [('Tests `tests/`', 1, 'yes')], [], []))

    def test_compare_uses_the_point_where_the_branch_started(self):
        with tempfile.TemporaryDirectory() as folder:
            def git(*args):
                subprocess.run(('git', '-c', 'user.name=t', '-c', 'user.email=t@example.com', *args),
                               cwd=folder, check=True, capture_output=True)

            def commit(name, text):
                (Path(folder) / name).write_text(text)
                git('add', '.')
                git('commit', '-m', name)

            git('init', '-b', 'main')
            commit('shared', '1\n')
            commit('CHANGELOG.md', '1\n')
            git('switch', '-c', 'topic')
            commit('shared', '2\n')
            commit('CHANGELOG.md', '2\n')
            commit('own', '1\n')
            git('switch', 'main')
            commit('shared', '3\n')
            commit('CHANGELOG.md', '3\n')
            files, size, behind, overlap = compare('main', 'topic', cwd=folder)
        self.assertEqual(files, ['CHANGELOG.md', 'own', 'shared'])
        self.assertIn('3 files changed', size)
        self.assertEqual((behind, overlap), (2, ['shared']))


if __name__ == '__main__':
    unittest.main()
