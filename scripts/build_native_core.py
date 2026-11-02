"""Build the optional CPython C++17 extension without third-party packages.

Production probability evaluation requires this extension. No network access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cxx',default='clang++')
    p.add_argument('--sanitize',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'core/magna')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    native=ROOT/'core/magna/native'
    # Preserve translation-unit order across responsibility-directory moves.
    units=sorted(native.rglob('*.cpp'),key=lambda p:(p.name,str(p)))
    sources=[*units,*sorted(native.rglob('*.hpp'))]
    module='_native'
    output=a.output_dir/(module+sysconfig.get_config_var('EXT_SUFFIX'))
    source_hashes={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    flags=['-std=c++17','-O3','-DNDEBUG','-Wall','-Wextra','-Wno-missing-field-initializers','-fPIC','-shared']
    if a.sanitize:flags=['-std=c++17','-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-fPIC','-shared']
    flags+=['-pthread']
    if sys.platform=='darwin':flags+=['-undefined','dynamic_lookup']
    start=datetime.now().astimezone().isoformat()
    with tempfile.TemporaryDirectory(prefix='.native-build-',dir=a.output_dir) as temporary:
        candidate=Path(temporary)/output.name
        cmd=[a.cxx,*flags,'-I'+sysconfig.get_path('include'),*map(str,units),'-o',str(candidate)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
        if not r.returncode:
            current={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
            if current!=source_hashes:
                raise RuntimeError('native sources changed during compilation; existing extension retained')
            candidate.replace(output)
    result=dict(started_at=start,finished_at=datetime.now().astimezone().isoformat(),command=cmd,
        compiler=subprocess.check_output([a.cxx,'--version'],text=True),python=sys.version,platform=platform.platform(),
        source_sha256=source_hashes,
        exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,linux_tested=sys.platform.startswith('linux'),
        binary_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if not r.returncode else None)
    (a.output_dir/'native-build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(shlex.join(cmd));print(r.stderr)
    if r.returncode:raise SystemExit(r.returncode)
    print(output)
if __name__=='__main__':main()
