"""Migrate paired Parquets without invoking ACTS, Geant4, or a truth matcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from scripts.postprocessing.utils import paired_schema as schema
except ModuleNotFoundError as e:
    if e.name != 'scripts': raise
    from utils import paired_schema as schema


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def counts(h, particles):
    n=len(h); known=particles['particle_id'][0].values
    if pc.count_distinct(known).as_py()!=len(known): raise ValueError('Duplicate particle IDs')
    flat=pc.list_flatten(h['particle_ids']); parents=pc.list_parent_indices(h['particle_ids']).to_numpy()
    if pc.any(pc.invert(pc.is_in(flat,value_set=known))).as_py(): raise ValueError('Missing contributor particle ID')
    labels=h['particle_id']; complete=h['particle_ids_complete'].to_numpy()
    if pc.any(pc.invert(pc.is_in(pc.drop_null(labels),value_set=known))).as_py(): raise ValueError('Missing scalar particle ID')
    nids=pc.list_value_length(h['particle_ids']).to_numpy(); missing=pc.list_value_length(h['unresolved_simhit_ids']).to_numpy()
    if not np.array_equal(complete,(missing==0)&(pc.list_value_length(h['simhit_ids']).to_numpy()>0)):
        raise ValueError('Invalid completeness flags')
    valid=pc.is_valid(labels).to_numpy()
    if np.any(valid & (~complete | (nids!=1))): raise ValueError('Invalid scalar truth semantics')
    legacy=h['converted_particle_id']; matches=np.zeros(n,bool)
    if len(flat):
        equal=pc.fill_null(pc.equal(flat,pc.take(legacy,pa.array(parents))),False).to_numpy()
        matches[parents[equal]]=True
        scalar_matches=np.zeros(n,bool)
        eq=pc.fill_null(pc.equal(flat,pc.take(labels,pa.array(parents))),False).to_numpy()
        scalar_matches[parents[eq]]=True
        if np.any(valid & ~scalar_matches): raise ValueError('Scalar absent from contributors')
    return dict(measurements=n,particles=len(known),
        simhit_links=int(pc.sum(pc.list_value_length(h['simhit_ids'])).as_py() or 0),
        multi_contributor_measurements=int((pc.list_value_length(h['simhit_ids']).to_numpy()>1).sum()),
        multi_particle_measurements=int((nids>1).sum()),null_particle_labels=labels.null_count,
        incomplete_truth_measurements=int((~complete).sum()),unresolved_simhit_links=int(missing.sum()),
        entirely_unresolved_measurements=int(((~complete)&(nids==0)).sum()),
        converted_label_disagreements=int((complete & pc.is_valid(legacy).to_numpy() & ~matches).sum()))


def converted_file(work, campaign, dataset, version):
    files=list((work/'paired/_conversion_results'/campaign/dataset/version/'parquet/reco/tracker_hits').glob('*.parquet'))
    if len(files)!=1: raise ValueError(f'Expected one retained conversion in {work}: {files}')
    return files[0]


def source_work(source, shard, production):
    pilot=json.loads((source/shard['provenance']/'pilot.json').read_text())
    rel=Path(pilot['pileup_reservation']['output']).relative_to('/output')
    candidates=[production/'output'/rel,Path('/tmp')/f'colliderml-paired-{os.getuid()}'/'output'/rel]
    for path in candidates:
        if (path/'paired/_conversion_results').is_dir(): return path
    raise ValueError(f'Retained converter outputs absent; checked {candidates}')


def migrate_sample(paths, converted, old_report, destination, context, offset, events):
    """Write and reread one shard. Peak memory is one event plus Arrow batches."""
    destination=Path(destination);destination.mkdir(parents=True)
    md=schema.metadata(context,offset,events)
    h_iter=iter(schema.iter_events(paths['hits']))
    c_iter=iter(pq.ParquetFile(converted).iter_batches(batch_size=1))
    p_iter=iter(pq.ParquetFile(paths['particles']).iter_batches(batch_size=1))
    totals={};maximum=0.; started=time.time()
    with pq.ParquetWriter(destination/'hits.parquet',schema.HIT_SCHEMA.with_metadata(md),compression='snappy') as writer, \
         pq.ParquetWriter(destination/'particles.parquet',schema.particle_schema(pq.read_schema(paths['particles']),md),compression='snappy') as pw:
        for local in range(events):
            old_event,h=next(h_iter); converted_batch=next(c_iter); particles=pa.Table.from_batches([next(p_iter)])
            if old_event!=local or converted_batch.column('event_id')[0].as_py()!=local or particles['event_id'][0].as_py()!=local:
                raise ValueError('Missing, duplicate or out-of-order source events')
            t, delta=schema.hit_event(h,pa.Table.from_batches([converted_batch]),offset+local,md)
            maximum=max(maximum,delta); new_h=schema.unnest(t)
            # Prove copied truth and local coordinates unchanged, independently of matching.
            for name in schema.EXTRA_FIELDS:
                if not schema.array_equal(h[name].combine_chunks(),new_h[name].combine_chunks()):
                    raise ValueError(f'Changed existing hit feature {name}')
            stats=counts(new_h,particles)
            for k,v in stats.items(): totals[k]=totals.get(k,0)+v
            writer.write_table(t,row_group_size=1)
            pp=schema.rewrite_particles(particles,offset,md)
            for name in particles.column_names:
                if name!='event_id' and not schema.array_equal(particles[name].combine_chunks(),pp[name].combine_chunks()):
                    raise ValueError(f'Changed particle feature {name}')
            pw.write_table(pp,row_group_size=1)
            if local%10==0: print(f"{context['run']} {context['campaign']}: hits {local+1}/{events}",flush=True)
    # Keep quantization metadata byte-for-byte; add schema/context metadata separately.
    old_cell_schema=pq.read_schema(paths['cells'])
    cell_md=schema.metadata(context,offset,events,old_cell_schema.metadata)
    fields=[pa.field('event_id' if f.name=='local_event' else f.name,f.type) for f in old_cell_schema]
    cell_schema=pa.schema(fields,metadata=cell_md)
    new_h_iter=iter(schema.iter_events(destination/'hits.parquet'))
    cells=0
    with pq.ParquetWriter(destination/'cells.parquet',cell_schema,compression='snappy') as cw:
        for local,old in schema.iter_events(paths['cells']):
            if local>=events: break
            ev,h=next(new_h_iter)
            if ev!=offset+local: raise ValueError('Cell/hit event mismatch')
            mid=old['measurement_id'].to_numpy(); geom=h['geometry_id'].to_numpy()
            if len(mid) and (mid.max()>=len(h) or not np.array_equal(geom[mid],old['geometry_id'].to_numpy())):
                raise ValueError('Orphan cell or wrong geometry')
            if np.any(np.bincount(mid.astype(np.int64),minlength=len(h))==0): raise ValueError('Hit without cells')
            for b in old.to_batches(max_chunksize=131072):
                source=pa.Table.from_batches([b]); new=schema.rewrite_cells(source,offset,context['version'],cell_md)
                for name in ('measurement_id','geometry_id','channel0','channel1','value','pixel_code'):
                    if not schema.array_equal(source[name].combine_chunks(),new[name].combine_chunks()): raise ValueError('Changed cell values')
                cw.write_table(new);cells+=len(new)
            if local%10==0: print(f"{context['run']} {context['campaign']}: cells {local+1}/{events}",flush=True)
    totals['cells']=cells
    if events==old_report['events']:
        for k,v in totals.items():
            if k in old_report and v!=old_report[k]: raise ValueError(f'{k}: {v} != source report {old_report[k]}')
    report={**old_report,**context,**totals,'events':events,'local_events':list(range(events)),
        'schema_version':schema.SCHEMA_VERSION,'event_id_start':offset,'event_id_stop_exclusive':offset+events,
        'source_event_mapping':'source_local_event = event_id - event_id_start',
        'maximum_xyz_roundtrip_difference_mm':maximum,'orphan_cells':0,'missing_particle_ids':0,
        'hit_layout':'one event row, aligned measurement lists','migration_validation_passed':True,
        'migration_seconds':time.time()-started,'subset_trial':events!=old_report['events'],'source_event_count':old_report['events'],'legacy_truth_semantics':'Copied from retained production converter; not complete contributor truth'}
    # Original diagnostic examples identify source-local events; retain them explicitly as such.
    report['source_converted_label_disagreement_examples']=old_report.get('converted_label_disagreement_examples',[])
    report['converted_label_disagreement_examples']=[{**{k:v for k,v in e.items() if k!='local_event'},
        'event_id':offset+e['local_event']} for e in old_report.get('converted_label_disagreement_examples',[])]
    # Reread output independently, comparing values against inputs, not just writer buffers.
    verify_sample(paths,converted,destination,context,offset,events,totals)
    write_json(destination/'report.json',report)
    return report


def verify_sample(paths,converted,dest,context,offset,events,expected):
    totals={}; originals=iter(schema.iter_events(paths['hits'])); conversions=iter(pq.ParquetFile(converted).iter_batches(batch_size=1))
    particles=iter(pq.ParquetFile(dest/'particles.parquet').iter_batches(batch_size=1))
    source_particles=iter(pq.ParquetFile(paths['particles']).iter_batches(batch_size=1))
    for i,(event,h) in enumerate(schema.iter_events(dest/'hits.parquet')):
        local,old=next(originals); legacy=pa.Table.from_batches([next(conversions)])
        if local!=i or event!=offset+i: raise ValueError('Output event range differs')
        wanted,_=schema.hit_event(old,legacy,event,schema.metadata(context,offset,events))
        for name in wanted.column_names:
            a=wanted[name][0].values if name!='event_id' else wanted[name].combine_chunks()
            b=h[name].combine_chunks() if name!='event_id' else pa.array([event],pa.uint32())
            if not schema.array_equal(a,b): raise ValueError(f'Reread hit mismatch: {name}')
        p=pa.Table.from_batches([next(particles)]); original=pa.Table.from_batches([next(source_particles)])
        if p['event_id'][0].as_py()!=event: raise ValueError('Particle event mismatch')
        for name in original.column_names:
            if name!='event_id' and not schema.array_equal(original[name].combine_chunks(),p[name].combine_chunks()): raise ValueError('Reread particle differs')
        for k,v in counts(h,p).items(): totals[k]=totals.get(k,0)+v
    cell_source=iter(schema.iter_events(paths['cells'])); new_hits=iter(schema.iter_events(dest/'hits.parquet'));total_cells=0
    for event,new in schema.iter_events(dest/'cells.parquet'):
        local,old=next(cell_source);hev,h=next(new_hits)
        if event!=offset+local or event!=hev: raise ValueError('Reread cells event mismatch')
        for name in ('measurement_id','geometry_id','channel0','channel1','value','pixel_code'):
            if not schema.array_equal(old[name].combine_chunks(),new[name].combine_chunks()): raise ValueError('Reread cells differ')
        mids=new['measurement_id'].to_numpy();geo=h['geometry_id'].to_numpy()
        if len(mids) and (mids.max()>=len(h) or not np.array_equal(geo[mids],new['geometry_id'].to_numpy())): raise ValueError('Reread orphan cells')
        total_cells+=len(new)
    totals['cells']=total_cells
    if totals!=expected: raise ValueError(f'Reread totals differ: {totals} / {expected}')
    if pq.read_metadata(dest/'hits.parquet').num_rows!=events or pq.read_metadata(dest/'particles.parquet').num_rows!=events:
        raise ValueError('Nested event row counts differ')
    if (pq.read_schema(paths['cells']).metadata or {}).get(b'colliderml.cell_quantization')!=(pq.read_schema(dest/'cells.parquet').metadata or {}).get(b'colliderml.cell_quantization'):
        raise ValueError('Quantization metadata changed')


def migrate(args):
    source=args.source.resolve(); output=args.output.resolve(); work=args.work_root.resolve()
    stage=work.with_name(work.name+'.partial')
    if output.exists() or work.exists() or stage.exists(): raise ValueError('Destination/work exists; choose a fresh path')
    old=json.loads((source/'manifest.json').read_text()); records=[];inputs={source/'manifest.json':schema.sha256(source/'manifest.json')};offset=0
    for shard in old['shards']:
        n=min(args.limit_events or shard['events_per_sample'],shard['events_per_sample'])
        records.append({**shard,'events_per_sample':n,'event_id_start':offset,'event_id_stop_exclusive':offset+n,
                        'files':{},'source_work':str(source_work(source,shard,args.production_root))})
        # Full migrations preserve the source shard allocation; trial shards may have gaps.
        offset+=shard['events_per_sample']
    schema.validate_ranges(records)
    estimated=sum((source/f['path']).stat().st_size for s in old['shards'] for tables in s['files'].values() for f in tables.values())
    if shutil.disk_usage(work.parent).free<2*estimated: raise ValueError('Insufficient staging space (require twice source Parquet size)')
    stage.mkdir(parents=True)
    code_snapshot = {Path(__file__).resolve():schema.sha256(__file__),Path(schema.__file__).resolve():schema.sha256(schema.__file__)}
    code_dir=stage/'provenance/migration_code';code_dir.mkdir(parents=True)
    for path in code_snapshot: shutil.copy2(path,code_dir/path.name)
    for original,record in zip(old['shards'],records):
        run=record['run']; prov=stage/f'provenance/run-{run:05d}';prov.mkdir(parents=True)
        shutil.copytree(source/original['provenance'],prov/'source')
        record['provenance']=f'provenance/run-{run:05d}'
        for sample in old['samples']:
            paths={k:source/v['path'] for k,v in original['files'][sample].items()}
            old_report=json.loads((source/original['provenance']/(sample+'_report.json')).read_text())
            conv=converted_file(Path(record['source_work']),old_report['campaign'],old_report['dataset'],old_report['version'])
            for path in [*paths.values(),conv,source/original['provenance']/(sample+'_report.json')]:
                inputs[path]=schema.sha256(path)
            for k,v in original['files'][sample].items():
                if inputs[paths[k]]!=v['sha256']: raise ValueError(f'Source checksum differs: {paths[k]}')
            dest=stage/f'run-{run:05d}/paired'/sample
            context={k:old_report[k] for k in schema.CONTEXT};context['version']=args.version
            report=migrate_sample(paths,conv,old_report,dest,context,record['event_id_start'],record['events_per_sample'])
            write_json(prov/(sample+'_report.json'),report)
            record['files'][sample]={k:{'source_relative':str((dest/(k+'.parquet')).relative_to(stage)),
                'sha256':schema.sha256(dest/(k+'.parquet')),'rows':pq.read_metadata(dest/(k+'.parquet')).num_rows}
                for k in ('hits','cells','particles')}
        try: record['parquet_source_relative']=str((work/f'run-{run:05d}/paired').relative_to(args.production_root.resolve()))
        except ValueError: record['parquet_source_relative']=None
    print('Rechecking all source checksums...',flush=True)
    for path,expected in inputs.items():
        if schema.sha256(path)!=expected: raise ValueError(f'Source changed: {path}')
    for path,digest in code_snapshot.items():
        if schema.sha256(path)!=digest:raise ValueError('Migration code changed during execution')
    receipt={'schema_version':schema.SCHEMA_VERSION,'passed':True,'source_sha256':{str(p):v for p,v in inputs.items()},
             'migration_code_sha256':{str(Path(__file__)):schema.sha256(__file__),str(Path(schema.__file__)):schema.sha256(schema.__file__)},
             'source_collection':str(source),'version':args.version,'events_per_sample':sum(r['events_per_sample'] for r in records),'shards':records}
    write_json(stage/'migration_validation.json',receipt)
    stage.rename(work)
    # Publish a separate collection; raw migration outputs remain for incremental backup.
    pub=output.with_name(output.name+'.partial')
    if pub.exists(): raise ValueError('Publication staging exists')
    pub.mkdir(parents=True)
    shutil.copytree(work/'provenance',pub/'provenance')
    for index,s in enumerate(records):
        for sample,tables in s['files'].items():
            for kind,r in tables.items():
                name=f'higgs_portal_{sample}_{kind}'
                relative=Path(name)/'data'/name/f'train-{index:05d}-of-{len(records):05d}.parquet'
                dest=pub/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                src=work/r.pop('source_relative')
                try: os.link(src,dest)
                except OSError: shutil.copy2(src,dest)
                if schema.sha256(dest)!=r['sha256']: raise ValueError('Publication copy checksum differs')
                r['path']=str(relative)
    manifest={**old,'format':'colliderml-homebrew-shards-v2','schema_version':schema.SCHEMA_VERSION,
              'version':args.version,'events_per_sample':receipt['events_per_sample'],'event_key':['campaign','dataset','version','event_id'],
              'particles_event_key':'event_id is dataset-wide; same ID in pu0/pu200 does not imply matched physics',
              'source_collection':str(source),'shards':records}
    write_json(pub/'migration_validation.json',receipt)
    write_json(pub/'manifest.json',manifest)
    pub.rename(output)
    print(f'PASS: migrated collection {output}',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/mnt/data2/data/collider_ml/homebrew/higgs_portal_v2_cells'))
    p.add_argument('--production-root',type=Path,default=Path('/mnt/data2/data/collider_ml/production'))
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--version',default='v2_cells_compat1')
    p.add_argument('--limit-events',type=int,help='Trial only: first N source events from each shard')
    a=p.parse_args()
    if a.limit_events is not None and a.limit_events<1:p.error('--limit-events must be positive')
    migrate(a)
