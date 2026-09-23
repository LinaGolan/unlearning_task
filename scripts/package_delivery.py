"""Package explicit source and evidence allowlists; never include environments or credentials."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
DIST.mkdir(exist_ok=True)
EXCLUDED = {'__pycache__', '.ipynb_checkpoints', '.git', '.venv', '.cache'}

def files_in(folder):
    for p in sorted((ROOT / folder).rglob('*')):
        if p.is_file() and not EXCLUDED.intersection(p.relative_to(ROOT).parts) and not p.name.startswith(('.env', '~$')) and p.suffix not in ('.pyc','.pyo'):
            yield p

def package(name, pairs):
    manifest = {}
    target = DIST / name
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as z:
        for source, relative in sorted(pairs, key=lambda x: str(x[1])):
            relative = str(relative).replace('\\','/')
            if relative in manifest:
                raise ValueError('Duplicate entry ' + relative)
            content = source.read_bytes()
            manifest[relative] = hashlib.sha256(content).hexdigest()
            z.writestr(relative, content, compress_type=ZIP_STORED if source.suffix=='.zip' else ZIP_DEFLATED)
        z.writestr('DELIVERY_MANIFEST.json',json.dumps({'files':manifest},indent=2)+'\n')
    with ZipFile(target) as z:
        assert z.testzip() is None
        for relative, digest in manifest.items():
            assert hashlib.sha256(z.read(relative)).hexdigest() == digest
    return {'name':name,'files':len(manifest),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}

source = []
for folder in ('src','tests','configs','notebooks','report/evidence','report/figures'):
    source.extend((p,p.relative_to(ROOT)) for p in files_in(folder))
for name in ('build_colab_bundle.py','build_stage2_notebook.py','build_stage3_bundle.py','build_stage4_bundle.py','build_stage5_bundle.py','build_stage6_bundle.py','package_delivery.py','plot_final_report.py','stage2_diagnostic_colab_cell.py','stage2_prefill_colab_cell.py','verify_stage5_results.py','verify_stage6_results.py'):
    source.append((ROOT/'scripts'/name,Path('scripts')/name))
for name in ('README.md','PLAN.md','REPRODUCE.md','ASSIGNMENT_COVERAGE.md','DIAGNOSTIC.md','pyproject.toml','.gitignore','.gitattributes','requirements-colab.txt','requirements-stage4-colab.txt'):
    source.append((ROOT/name,name))
for p in sorted(ROOT.glob('STAGE[1-6]*.md')):
    source.append((p,p.name))

evidence = []
folders = {
    'data/prepared/full':'data/prepared/full',
    'outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill':'outputs/stage2/prefill',
    'outputs/stage2/verified_2026-09-20/diagnostic/outputs':'diagnostic/outputs',
    'outputs/stage3/verified_2026-09-20/outputs/stage3/20260920T160702Z-b27cdc':'outputs/stage3/check',
    'outputs/stage4/verified_2026-09-20/outputs/stage4/full':'outputs/stage4/full',
}
for folder, destination in folders.items():
    for p in files_in(folder):
        evidence.append((p,Path(destination)/p.relative_to(ROOT/folder)))
for stage in (2,3,4,5,6):
    date = '2026-09-20' if stage < 5 else '2026-09-21'
    path=ROOT/('outputs/stage%d/verified_%s/verification.json'%(stage,date))
    evidence.append((path,'verification/stage%d.json'%stage))
for stage in (5,6):
    evidence.append((ROOT/('outputs/stage%d/verified_2026-09-21/original_results.zip'%stage),'archives/stage%d-results.zip'%stage))
for archive_source, destination in (
    ('outputs/stage1/colab_verified_2026-09-19/stage1-results.zip','archives/stage1-results.zip'),
    ('outputs/stage2/verified_2026-09-20/prefill/original-download.zip','archives/stage2-prefill-results.zip'),
    ('outputs/stage2/verified_2026-09-20/diagnostic/original-download.zip','archives/stage2-diagnostic-results.zip'),
    ('outputs/stage3/verified_2026-09-20/original-download.zip','archives/stage3-results.zip'),
    ('outputs/stage4/verified_2026-09-20/original-download.zip','archives/stage4-results.zip'),
):
    evidence.append((ROOT/archive_source,destination))
evidence.append((ROOT/'REPRODUCE.md','REPRODUCE.md'))
evidence.append((ROOT/'ASSIGNMENT_COVERAGE.md','ASSIGNMENT_COVERAGE.md'))
result = [package('unlearning-source.zip',source),package('unlearning-verified-evidence.zip',evidence)]
(DIST/'delivery_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
