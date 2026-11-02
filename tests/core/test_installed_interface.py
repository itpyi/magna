"""Wheel acceptance tests. Set MAGNA_INSTALLED_PYTHON to an isolated install.

The subprocess uses -I and a temporary working directory, so the repository's
source paths cannot satisfy missing wheel modules or numerical data.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALLED_PYTHON = os.environ.get("MAGNA_INSTALLED_PYTHON")


@unittest.skipUnless(INSTALLED_PYTHON, "set MAGNA_INSTALLED_PYTHON to test a built wheel")
class InstalledInterfaceTests(unittest.TestCase):
    def test_installed_cuda_sources_without_optional_dependencies(self):
        self.python('''
import importlib.abc, sys
class NoOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'cupy'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoOptionalImports())
from magna import _native, parse_circuit, run
from magna._cuda_source import _read_cuda_source
for name in ('noise_sampling.cu', 'probability_queries.cu', 'statistics.cu'):
    source = _read_cuda_source(name)
    assert '__global__' in source, name
for name in ('make_cuda_experiment', 'make_cuda_queries', 'cuda_noise_metadata'):
    assert hasattr(_native, name), name
assert run(parse_circuit('I 0'), shots=4, seed=9)['status'] == 'complete'
''')

    def test_portable_directory_with_local_decoder_and_universal_runner(self):
        shutil.copytree(ROOT / 'examples/dk', self.directory / 'task',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'scripts/run_task.py', self.directory / 'run_task.py')
        process = subprocess.run([INSTALLED_PYTHON, '-I', 'run_task.py', 'task',
            '--shots', '4', '--seed', '7', '--output', 'result.json'],
            cwd=self.directory, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads((self.directory / 'result.json').read_text())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['completed'], 4)

    def test_installed_record_success_entry(self):
        self.python('''
from pathlib import Path
import json
from magna import enumerate_errors, parse_circuit, run
circuit = parse_circuit('X_ERROR(1/5) 0\\nM(1/7) 0\\nOBSERVABLE_INCLUDE(0) rec[-1]')
result = enumerate_errors(circuit, task={'error_observables': [0]}, max_weight=2)
assert result['totals']['error'] == 2/7, result
Path('record.magna').write_text(circuit.to_text())
Path('record.json').write_text(json.dumps({'error_observables': [0]}))
result = run('record.magna', task='record.json', shots=8, seed=3)
assert result['status'] == 'complete', result
''')
        result = subprocess.run([INSTALLED_PYTHON, '-I', '-m', 'magna.file_api',
            'enumerate', 'record.magna', '--task', 'record.json', '--max-weight', '2', '--json'],
            cwd=self.directory, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['totals']['error'], 2/7)

    def test_installed_history_progress_and_pilot(self):
        self.python('''
from contextlib import redirect_stderr
from io import StringIO
from magna import compare_samplers, parse_circuit, run
circuit = parse_circuit('X_ERROR(1/2) 0')
target = parse_circuit('I 0')
progress = StringIO()
with redirect_stderr(progress):
    result = run(circuit, ideal_state=target, shots=32, seed=13, progress_every=16)
assert result['status'] == 'complete', result
assert result['history_sampler'] == 'affine_support'
assert result['efficiency']['kappa'] == 1
assert progress.getvalue().count('kappa=') == 2
pilot = compare_samplers(circuit, ideal_state=target, pilot_shots=32, seed=13)
assert pilot['status'] == 'complete', pilot
assert not pilot['automatic_switching']
assert pilot['gamma'] > 0
''')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="magic-installed-interface-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for source in (ROOT / "tests/fixtures/basic").iterdir():
            if source.is_file():
                shutil.copy2(source, self.directory / source.name)

    def python(self, code):
        result = subprocess.run(
            [INSTALLED_PYTHON, "-I", "-c", code], cwd=self.directory,
            text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_installed_conditional_noise_fixed_reference_networks(self):
        self.python('''
from magna import compile_circuit, enumerate_errors, parse_circuit, run
circuit = parse_circuit('H 0\\nM 0\\nX_ERROR(1/3) rec[-1] 1\\nM 1\\nX_ERROR(1/2) rec[-1] 2')
kwargs = dict(task={'outputs': [2]}, ideal_state=parse_circuit('I 0'))
compiled = compile_circuit(circuit, **kwargs)
before = compiled._execution.state()
assert before['conditional_noise'] and before['sampled_history']
assert before['reference_network_preparations'] == 3
assert run(compiled, shots=64, seed=3)['status'] == 'complete'
after = compiled._execution.state()
assert after['compilations'] == before['compilations']
assert after['reference_network_preparations'] == 3
assert after['reference_network_resource_constructions'] == after['reference_network_coordinate_constructions'] == 0
exact = enumerate_errors(circuit, max_weight=2, **kwargs)
assert exact['status'] == 'complete'
assert exact['totals']['acceptance'] == 1
assert exact['totals']['error'] == 1/12
''')

    def test_wheel_global_query_and_legacy_api_without_protocol_paths(self):
        self.python('''
import importlib.abc, json, math, sys
from pathlib import Path
class NoProtocolImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'cases', 'scripts', 'results', 'magic15', 'magic422',
            'magic_cultivation', 'magic_jit', 'magic_multi', 'magic_surface', 'magic_twisted'
        }:
            raise AssertionError('unexpected protocol import: ' + fullname)
sys.meta_path.insert(0, NoProtocolImports())
from importlib.resources import files
import magna
from magna import _native, load_circuit, compile_circuit, run
from magna.operations import InstrumentIR, Simulator, SampleAssignment, StabilizerKet
assert 'site-packages' in magna.__file__, magna.__file__
assert files('magna').joinpath('data/phase_decompositions.json').is_file()
for retired in ('skeleton', 'contraction', 'phase_quotient', 'phase_trilinear',
                'phase', 'signature', 'database', 'virtual_feedback', 'resolved_feedback'):
    assert not files('magna').joinpath(retired + '.py').is_file(), retired
assert not hasattr(_native, 'make_plan')
assert not hasattr(_native, 'evaluate_batch')
task = json.loads(Path('t_interference.task.json').read_text())
circuit = load_circuit('t_interference.magna')
compiled = compile_circuit(circuit, task=task)
result = run(circuit, task=task, shots=4, seed=7)
assert result['status'] == 'complete', result
assert result['completed'] == 4, result
assert math.isclose(result['estimates']['acceptance'], (2 + math.sqrt(2)) / 4), result
legacy = InstrumentIR(inputs=(0,), outputs=(0,), initial=StabilizerKet.positive(1),
    operations=({'id':'h', 'kind':'H', 'qubits':[0]},
                {'id':'m', 'kind':'MZ', 'qubits':[0], 'record':'m'}))
assert math.isclose(float(Simulator(legacy).probability(SampleAssignment(true={'m':0})).real_decimal()), .5)
''')

    def test_installed_console_commands_and_seed_reproducibility(self):
        binary = Path(INSTALLED_PYTHON).parent / "magna"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        for command in ("validate", "inspect", "run"):
            args = [str(binary), command, "parity_feedback.magna"]
            if command != "validate":
                args += ["--task", "parity_feedback.task.json"]
            if command == "run":
                args += ["--shots", "8", "--seed", "19", "--output", "result.json"]
            process = subprocess.run(args, cwd=self.directory, env=env,
                                     text=True, capture_output=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads((self.directory / "result.json").read_text())
        self.assertEqual(result["completed"], 8)
        self.python('''
import json
from pathlib import Path
from magna import run
task = json.loads(Path('parity_feedback.task.json').read_text())
a = run('parity_feedback.magna', task=task, shots=8, seed=19)
b = run('parity_feedback.magna', task=task, shots=8, seed=19)
assert a['estimates'] == b['estimates'], (a,b)
''')

    def test_installed_external_decoder_binding(self):
        self.python('''
import json
from pathlib import Path
from magna import load_controllers, load_circuit, run
registry = load_controllers('controllers.json')
circuit = load_circuit('majority_feedback.magna', controllers=registry)
assert circuit.analysis.route == 'nonlinear'
task = json.loads(Path('majority_feedback.task.json').read_text())
result = run(circuit, task=task, shots=16, seed=7)
assert result['status'] == 'complete', result
assert result['completed'] == 16, result
assert result['estimates']['acceptance'] == 1, result
assert 0 <= result['estimates']['error'] <= 1, result
''')

    def test_installed_nonlinear_postselection_api_and_console(self):
        self.python('''
from magna import load_controllers, load_circuit, parse_circuit, enumerate_errors, run
registry = load_controllers('controllers.json')
circuit = load_circuit('nonlinear_postselection.magna', controllers=registry)
assert circuit.format_version == 'magna-0.1'
task = 'nonlinear_postselection.task.json'
result = enumerate_errors(circuit, task=task, max_weight=0)
assert result['status'] == 'complete', result
assert result['totals']['acceptance'] == .75, result
assert result['totals']['error'] == .25, result
assert result['totals']['moments'] == [.25], result
assert result['totals']['conditional_error_within_enumerated_sector'] == 1/3, result
a = run(circuit, task=task, shots=32, seed=7)
b = run(circuit, task=task, shots=32, seed=7)
assert a['status'] == 'complete', a
assert a['completed'] == 32, a
assert a['estimates'] == b['estimates'], (a,b)
assert 0 < a['estimates']['acceptance'] < 1, a
inverse = parse_circuit(circuit.to_text().replace('POSTSELECT ctrl[0]',
    'CONTROL[NOT] ctrl[0] -> ctrl[1]\\nPOSTSELECT ctrl[1]'), controllers=registry)
assert inverse.format_version == 'magna-0.1'
result = enumerate_errors(inverse, task=task, max_weight=0)
assert result['status'] == 'complete', result
assert result['totals']['acceptance'] == .25, result
assert result['totals']['error'] == .25, result
for bit in (0, 1):
    try:
        parse_circuit(circuit.to_text().replace('POSTSELECT ctrl[0]', f'POSTSELECT({bit}) ctrl[0]'),
                      controllers=registry)
    except ValueError as exc:
        assert 'POSTSELECT takes no arguments' in str(exc), exc
    else:
        raise AssertionError('parameterized POSTSELECT must be rejected')
''')
        binary = Path(INSTALLED_PYTHON).parent / 'magna'
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        for command in ('validate', 'inspect', 'enumerate', 'run'):
            args = [str(binary), command, 'nonlinear_postselection.magna',
                    '--controllers', 'controllers.json', '--task', 'nonlinear_postselection.task.json', '--json']
            if command == 'enumerate': args += ['--max-weight', '0']
            if command == 'run': args += ['--shots', '32', '--seed', '7']
            process = subprocess.run(args, cwd=self.directory, env=env,
                                     text=True, capture_output=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            result = json.loads(process.stdout)
            if command in ('validate', 'inspect'): self.assertEqual(result['format'], 'magna-0.1')
            if command == 'enumerate': self.assertEqual(result['totals']['acceptance'], .75)

    def test_getting_started_python_examples(self):
        document = (ROOT / "docs/software/getting-started.md").read_text()
        circuit = re.findall(r"```text\n(.*?)```", document, re.S)[0]
        (self.directory / "example.magna").write_text(circuit)
        examples = re.findall(r"```python\n(.*?)```", document, re.S)
        self.assertEqual(len(examples), 2)
        self.python("\n".join(examples))

    def test_three_file_evaluation_with_installed_native_engine(self):
        shutil.copytree(ROOT / 'tests/fixtures/terminal-recovery', self.directory / 'terminal')
        self.python('''
import importlib.abc, json, sys
from pathlib import Path
class NoProtocolImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cases','scripts','results','magic_twisted','magic_surface','magic_multi'}:
            raise AssertionError('unexpected research import: ' + fullname)
sys.meta_path.insert(0, NoProtocolImports())
from magna import load_controllers, run
from magna.ideal_state import IdealState
from magna.terminal_recovery import TerminalRecovery
assert 'site-packages' in sys.modules[IdealState.__module__].__file__
registry = load_controllers('terminal/controllers.json')
result = run('terminal/physical.magna', controllers=registry, task='terminal/task.json', shots=4, seed=7)
assert result['status'] == 'complete', result
assert result['estimates']['acceptance'] == 1, result
assert result['estimates']['error'] == 0, result
raw = run('terminal/physical.magna', ideal_state='terminal/ideal.magna')
assert raw['estimates']['error'] == 1, raw
from magna import parse_circuit
auxiliary = dict(circuit=parse_circuit('H 0 1 2 3 4\\nCCZ 0 3 4\\nCCZ 1 2 4\\nH 4'),
                 outputs=[0,1,2,3], postselection={4:0})
projected = run(parse_circuit('I 0 1 2 3'), ideal_state=auxiliary)
assert projected['status'] == 'complete', projected
assert abs(projected['estimates']['error']-.9) < 1e-12, projected
mixed = run(parse_circuit('I 0'), ideal_state=dict(circuit=parse_circuit('H 0\\nCX 0 1'), outputs=[1]))
assert mixed['estimates']['error'] == .5, mixed
''')
        binary = Path(INSTALLED_PYTHON).parent / 'magna'
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        process = subprocess.run([str(binary), 'run', 'terminal/physical.magna',
            '--controllers', 'terminal/controllers.json', '--ideal-state', 'terminal/ideal.magna',
            '--recovery', 'terminal/recovery.magna', '--shots', '2', '--json'],
            cwd=self.directory, env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        self.assertEqual(json.loads(process.stdout)['estimates']['error'], 0)

    def test_importance_sampling_guide_and_installed_cli(self):
        for source in (ROOT / 'tests/fixtures/importance-sampling').iterdir():
            if source.is_file(): shutil.copy2(source, self.directory / source.name)
        document = (ROOT / 'docs/software/sampling.md').read_text()
        examples = re.findall(r"```python\n(.*?)```", document, re.S)
        self.assertEqual(len(examples), 4)
        self.python('''
import importlib.abc, sys
class NoProtocolImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cases','scripts','results','magic_twisted','magic_surface','magic_multi'}:
            raise AssertionError('unexpected research import: ' + fullname)
sys.meta_path.insert(0, NoProtocolImports())
import magna
assert 'site-packages' in magna.__file__
''' + '\n'.join(examples[:3]) + '''
assert result['status'] == 'complete', result
assert result['sampling']['tilted_categories'] == 1, result
assert 0 < result['weights']['ess'] <= 1000, result
assert abs(result['estimates']['error'] - .001) < 4 * result['standard_errors']['error'], result
assert abs(result['estimates']['acceptance'] - 1) < 4 * result['standard_errors']['acceptance'], result
''')
        self.python(examples[3] + '''
assert result['status'] == 'complete', result
assert result['completed'] == 10000, result
assert result['workers'] == 4, result
''')
        binary = Path(INSTALLED_PYTHON).parent / 'magna'
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        for extra_args, workers in (([], 1), (['--workers', '4'], 4)):
            with self.subTest(workers=workers):
                process = subprocess.run([str(binary), 'run', 'physical.magna', '--task', 'task.json',
                    '--sampling', 'sampling.json', '--shots', '64', '--seed', '7', '--json', *extra_args],
                    cwd=self.directory, env=env, text=True, capture_output=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                result = json.loads(process.stdout)
                self.assertEqual(result['completed'], 64)
                self.assertEqual(result['sampling']['method'], 'importance')
                self.assertEqual(result['workers'], workers)

    def test_installed_low_weight_enumeration(self):
        self.python("""
from importlib.util import find_spec
from magna import enumerate_errors, parse_circuit
for name in ('adapters', 'protocol_decoders', 'migration_decoders', 'css_diagnostic'):
    assert find_spec('magna.' + name) is None, name
circuit = parse_circuit('X_ERROR(1/10) 0\\nM 0\\nDETECTOR rec[-1]\\nPOSTSELECT det[0]')
r = enumerate_errors(circuit, max_weight=1)
assert r['status'] == 'complete', r
assert abs(r['totals']['acceptance'] - .9) < 1e-12, r
assert r['totals']['covered_probability'] == 1, r
""")


if __name__ == "__main__":
    unittest.main()
