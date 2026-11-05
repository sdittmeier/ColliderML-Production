"""Read-only manifest, schema, event-range and checksum checks for a release."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scripts.postprocessing.utils import paired_schema as s
except ModuleNotFoundError as e:
    if e.name!='scripts':raise
    from utils import paired_schema as s


def validate(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    receipt=json.loads((root/'migration_validation.json').read_text())
    if not receipt['passed'] or m['schema_version']!=s.SCHEMA_VERSION:raise ValueError('Invalid completion/schema marker')
    s.validate_ranges(m['shards']);result={}
    for sample in m['samples']:
        seen=[];totals=dict(events=0,measurements=0,cells=0,particles=0,incomplete_truth_measurements=0,
                            entirely_unresolved_measurements=0,unresolved_simhit_links=0,converted_label_disagreements=0)
        for shard in m['shards']:
            expected=list(range(shard['event_id_start'],shard['event_id_stop_exclusive']))
            report=json.loads((root/shard['provenance']/(sample+'_report.json')).read_text())
            for kind,r in shard['files'][sample].items():
                path=root/r['path'];f=pq.ParquetFile(path)
                if s.sha256(path)!=r['sha256'] or f.metadata.num_rows!=r['rows']:raise ValueError(f'Checksum/rows differ: {path}')
                if (f.schema_arrow.metadata or {}).get(s.SCHEMA_KEY)!=s.SCHEMA_VERSION.encode():raise ValueError('Missing schema metadata')
                if 'local_event' in f.schema_arrow.names or any(n in f.schema_arrow.names for n in ('global_x','global_y','global_z')):raise ValueError('Obsolete aliases on disk')
                ctx=s.context(f.schema_arrow)
                if ctx['event_id_start']!=expected[0] or ctx['run']!=shard['run']:raise ValueError('Wrong context metadata')
                if kind in ('hits','particles'):
                    ids=pq.read_table(path,columns=['event_id'])['event_id'].to_pylist()
                    if ids!=expected:raise ValueError('Wrong nested event IDs')
                if kind=='hits':
                    if not f.schema_arrow.equals(s.HIT_SCHEMA,check_metadata=False):raise ValueError('Wrong hit schema')
                    for b in f.iter_batches(batch_size=1):
                        n=len(b.column('measurement_id')[0].values)
                        if any(len(b.column(i)[0].values)!=n for i,fld in enumerate(b.schema) if fld.name!='event_id'):
                            raise ValueError('Misaligned nested lists')
                if kind=='cells':
                    ids=set()
                    for b in f.iter_batches(columns=['event_id']):ids.update(b.column('event_id').to_pylist())
                    if sorted(ids)!=expected:raise ValueError('Wrong cell event IDs')
            seen+=expected
            for k in totals:totals[k]+=report[k]
        if seen!=list(range(m['events_per_sample'])):raise ValueError('Collection has event gaps or duplicates')
        result[sample]=totals
    print(json.dumps({'passed':True,'schema_version':s.SCHEMA_VERSION,'samples':result},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    validate(p.parse_args().root)
