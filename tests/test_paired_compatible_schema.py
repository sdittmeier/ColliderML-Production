"""Schema migration preserves physical data and makes event identity explicit."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.postprocessing import export_paired_clusters as exporter
from scripts.postprocessing import migrate_paired_schema as migration
from scripts.postprocessing.utils import paired_schema as schema


class CompatibleSchemaTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.context=dict(campaign='full_pileup',dataset='ttbar',version='compat1',run=4)
        self.md=schema.metadata(self.context,100,1)
        geom=(16<<56)|(2<<36)|(3<<8)
        self.hits=pa.Table.from_pylist([
            dict(campaign='full_pileup',dataset='ttbar',version='v1',run=4,local_event=0,
                measurement_id=i,geometry_id=geom,global_x=1.+i,global_y=2.,global_z=3.,
                local0=-.1,local1=.2,particle_id=4 if i==0 else None,converted_particle_id=4,
                particle_ids=[4] if i==0 else [4,5],particle_ids_complete=True,
                unresolved_simhit_ids=[],simhit_ids=[0] if i==0 else [1,2]) for i in range(2)],schema=exporter.HIT_SCHEMA)
        values={'x':[1.,2.],'y':[2.,2.],'z':[3.,3.],'volume_id':[16,16],
                'layer_id':[2,2],'surface_id':[3,3],'true_x':[1.,np.nan],
                'true_y':[2.,np.nan],'true_z':[3.,np.nan],'time':[.1,np.nan],'detector':[1,255]}
        self.converted=pa.table({'event_id':pa.array([0],pa.uint32()),**{
            k:pa.array([v],pa.list_(schema.LEGACY_FIELDS[k])) for k,v in values.items()}})
        self.particles=pa.table({'event_id':pa.array([0],pa.uint32()),
            'particle_id':pa.array([[4,5]],pa.list_(pa.uint64())),
            'perigee_d0':pa.array([[0.,np.nan]],pa.list_(pa.float32()))})
        self.cells=pa.Table.from_pylist([dict(campaign='full_pileup',dataset='ttbar',version='v1',run=4,
            local_event=0,measurement_id=i,geometry_id=geom,channel0=3+i,channel1=5,
            value=.012+i*.01,pixel_code=1+i) for i in range(2)],schema=exporter.CELL_SCHEMA)

    def test_established_names_types_and_nested_lists(self):
        table,delta=schema.hit_event(self.hits,self.converted,100,self.md)
        self.assertEqual(delta,0.)
        self.assertEqual(table.num_rows,1)
        self.assertEqual(table['event_id'].to_pylist(),[100])
        self.assertNotIn('global_x',table.column_names);self.assertNotIn('local_event',table.column_names)
        self.assertNotIn('run',table.column_names)
        for name,typ in schema.LEGACY_FIELDS.items():
            self.assertEqual(table.schema.field(name).type,pa.list_(typ))
            self.assertTrue(schema.array_equal(table[name][0].values,self.converted[name][0].values))
        self.assertEqual(table['particle_ids'].to_pylist(),[[[4],[4,5]]])
        p=self.root/'hits.parquet';pq.write_table(table,p)
        flat=schema.read_event(p,100)
        self.assertEqual(flat['run'].to_pylist(),[4,4])
        self.assertEqual(flat['event_id'].to_pylist(),[100,100])
        self.assertEqual(schema.source_event(p,100),0)
        with self.assertRaisesRegex(ValueError,'outside'):schema.source_event(p,99)

    def test_old_flat_reader_and_cell_quantization_preservation(self):
        p=self.root/'old.parquet';pq.write_table(self.hits,p)
        old=schema.read_event(p,0)
        self.assertIn('x',old.column_names);self.assertIn('event_id',old.column_names)
        md=schema.metadata(self.context,100,1,{b'colliderml.cell_quantization':b'{"scheme":"p99.9"}'})
        cells=schema.rewrite_cells(self.cells,100,'compat1',md)
        self.assertEqual(cells['event_id'].to_pylist(),[100,100])
        for name in ('value','pixel_code','channel0','channel1','measurement_id'):
            self.assertTrue(schema.array_equal(self.cells[name].combine_chunks(),cells[name].combine_chunks()))
        self.assertEqual(cells.schema.metadata[b'colliderml.cell_quantization'],b'{"scheme":"p99.9"}')

    def test_unequal_shards_ranges_and_overflow(self):
        self.assertEqual(schema.event_offset(103,29,2),103)
        schema.validate_ranges([dict(event_id_start=100,event_id_stop_exclusive=103),
                                dict(event_id_start=103,event_id_stop_exclusive=105)])
        with self.assertRaisesRegex(ValueError,'explicit'):schema.event_offset(None,4,100)
        for offset,events in ((-1,1),(2**32,1),(2**32-1,2)):
            with self.assertRaises(ValueError):schema.event_offset(offset,0,events)
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            schema.validate_ranges([dict(event_id_start=0,event_id_stop_exclusive=3),dict(event_id_start=2,event_id_stop_exclusive=4)])

    def test_broken_alignment_and_missing_columns(self):
        with self.assertRaisesRegex(ValueError,'required fields'):
            schema.hit_event(self.hits,self.converted.drop(['time']),100,self.md)
        with self.assertRaisesRegex(ValueError,'measurement IDs'):
            schema.hit_event(self.hits.take(pa.array([1,0])),self.converted,100,self.md)

    def test_empty_event_preserves_row_and_list_types(self):
        empty=self.converted.set_column(0,'event_id',pa.array([0],pa.uint32()))
        for name in schema.LEGACY_FIELDS:
            idx=empty.schema.get_field_index(name)
            empty=empty.set_column(idx,name,pa.array([[]],pa.list_(schema.LEGACY_FIELDS[name])))
        table,_=schema.hit_event(self.hits.slice(0,0),empty,100,self.md)
        self.assertEqual(table.num_rows,1)
        self.assertEqual(table['x'].to_pylist(),[[]])
        self.assertEqual(schema.unnest(table).num_rows,0)

    def test_entirely_unresolved_and_empty_truth_are_valid(self):
        h=self.hits.slice(0,1)
        for name,value in [('particle_id',None),('particle_ids',[]),('particle_ids_complete',False),('unresolved_simhit_ids',[0])]:
            h=h.set_column(h.schema.get_field_index(name),name,pa.array([value],h.schema.field(name).type))
        stats=migration.counts(schema.canonical_flat(h),self.particles)
        self.assertEqual(stats['entirely_unresolved_measurements'],1)
        self.assertEqual(stats['unresolved_simhit_links'],1)
        empty=schema.canonical_flat(h.slice(0,0))
        stats=migration.counts(empty,self.particles)
        self.assertEqual(stats['measurements'],0)
        self.assertEqual(stats['simhit_links'],0)

    def run_migration(self,broken=False):
        paths={k:self.root/(k+'.parquet') for k in ('hits','cells','particles')}
        cells=self.cells
        if broken:cells=cells.set_column(cells.schema.get_field_index('measurement_id'),'measurement_id',pa.array([99,1],pa.uint64()))
        for k,t in [('hits',self.hits),('cells',cells),('particles',self.particles)]:pq.write_table(t,paths[k])
        conv=self.root/'converted.parquet';pq.write_table(self.converted,conv)
        before={p:schema.sha256(p) for p in [*paths.values(),conv]}
        report={'events':1,'campaign':'full_pileup','dataset':'ttbar','version':'v1','run':4,
                'cell_quantization':{},'converted_label_disagreement_examples':[]}
        if broken:
            with self.assertRaisesRegex(ValueError,'Orphan cell'):
                migration.migrate_sample(paths,conv,report,self.root/'out',self.context,100,1)
        else:
            r=migration.migrate_sample(paths,conv,report,self.root/'out',self.context,100,1)
            self.assertTrue(r['migration_validation_passed']);self.assertEqual(r['measurements'],2)
            p=pq.read_table(self.root/'out/particles.parquet')
            self.assertEqual(p['event_id'].to_pylist(),[100])
            self.assertTrue(schema.array_equal(p['perigee_d0'].combine_chunks(),self.particles['perigee_d0'].combine_chunks()))
        self.assertEqual(before,{p:schema.sha256(p) for p in before})

    def test_full_migration_roundtrip_and_source_checksums(self):self.run_migration()
    def test_orphan_cell_fails_without_changing_inputs(self):self.run_migration(True)


if __name__=='__main__':unittest.main()
