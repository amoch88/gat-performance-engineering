"""Synthetic correctness and timing-accounting tests; not dataset performance."""
import itertools
import unittest
from core import *


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        cls.pilot=pilot()

    def graph(self):
        torch.manual_seed(9)
        # Includes parallel edges, isolated nodes before loops, and unsorted EIDs.
        g=dgl.graph(([3,0,2,1,0,2,2],[0,2,0,3,1,3,0]),num_nodes=6)
        g=dgl.add_self_loop(dgl.remove_self_loop(g))
        g.ndata['feat']=torch.randn(6,11)
        g.create_formats_();attach_ids(g)
        return g

    def test_pilot(self):
        self.assertEqual(len(self.pilot['policies']),7)
        self.assertIn('all',valid_formats(self.pilot))
        for r in self.pilot['policies']:
            self.assertEqual(r['after']['allowed_formats'],sorted(r['allowed']))
            if r['status']=='supported': self.assertTrue(r['numerical_equivalence'])
            else: self.assertTrue(r['error'])

    def test_matrix_and_original_eids_after_node_relabeling(self):
        g=self.graph();model=model_for(11)
        with torch.inference_mode(): ref=model(g,g.ndata['feat'])
        for n,e,f in configurations(valid_formats(self.pilot)):
            with self.subTest(node=n,edge=e,format=f):
                case=prepare(g,n,e,f)
                check=verify_prepared(case,g,model,ref)
                self.assertTrue(check['topology_and_eids'])
                if n!='original':
                    bad=case['x'].clone();case['x']=bad.roll(1,0)
                    with self.assertRaises(AssertionError): verify_prepared(case,g,model,ref)

    def test_corrupt_edge_id_map_is_rejected(self):
        g=self.graph();model=model_for(11)
        with torch.inference_mode(): ref=model(g,g.ndata['feat'])
        case=prepare(g,'rcmk','original','all')
        case['graph'].edata['joint_eid']=case['graph'].edata['joint_eid'].roll(1)
        with self.assertRaises(AssertionError): verify_prepared(case,g,model,ref)

    def test_timing_accounting(self):
        g=self.graph();case=prepare(g,'rcmk','original','all');model=model_for(11)
        block=timed_block(case,model,1,3)
        for f,r,c in zip(block['forward_ms']['samples'],block['restoration_ms']['samples'],block['combined_ms']['samples']):
            self.assertAlmostEqual(f+r,c,places=10)
            self.assertGreater(f,0);self.assertGreater(r,0)

    def test_exact_duplicate_signature(self):
        g=dgl.add_self_loop(dgl.graph(([],[]),num_nodes=4))
        g.ndata['feat']=torch.arange(12,dtype=torch.float32).reshape(4,3)
        g.create_formats_();attach_ids(g);model=model_for(3)
        with torch.inference_mode(): ref=model(g,g.ndata['feat'])
        a=verify_prepared(prepare(g,'original','original','all'),g,model,ref)
        b=verify_prepared(prepare(g,'degree','original','all'),g,model,ref)
        self.assertEqual(a['signature'],b['signature'])


class AnalysisTests(unittest.TestCase):
    def rounds(self, interaction=False):
        formats=['all','coo+csc']
        keys=['/'.join(c) for c in configurations(formats)]
        factors=[{'original':1,'rcmk':.9,'degree':1.1},
                 {'original':1,'src':.9,'dst':1.1},{'all':1,'coo+csc':.95}]
        output=[]
        for index in range(4):
            trials=[]
            for key in keys:
                n,e,f=key.split('/')
                value=factors[0][n]*factors[1][e]*factors[2][f]
                if interaction and key=='degree/dst/coo+csc': value=.5
                trials.append({'key':key,'normalized':{m:{'relative_latency':value}
                    for m in ['forward_ms','combined_ms']}})
            output.append({'trials':trials,'aliases':{k:k for k in keys}})
        return output

    def test_multiplicative_null_interactions(self):
        from analysis import interactions
        for pair in interactions(self.rounds(),['all','coo+csc'],'combined_ms'):
            for row in pair['cells']:
                for cell in row: self.assertAlmostEqual(cell['geomean'],1.)

    def test_joint_vs_axis_selection(self):
        from analysis import select
        s=select(self.rounds(True),['all','coo+csc'],'combined_ms')
        self.assertEqual(s['joint'],'degree/dst/coo+csc')
        self.assertEqual(s['independent'],'rcmk/src/coo+csc')
        self.assertTrue(s['different'])

    def test_interval_and_ratio(self):
        from analysis import ratio_summary
        self.assertEqual(ratio_summary([1.,1.,1.])['ci95'],[1.,1.])
        self.assertAlmostEqual(ratio_summary([.5,2.])['geomean'],1.)


if __name__=='__main__': unittest.main()
