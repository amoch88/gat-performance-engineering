"""All four families: correctness tests, not real-dataset performance results."""
import unittest
from common import *
from native import Native
import index_dtype,traversal,layouts,scheduling
from orchestration import Pipeline

FAMILIES={'index_dtype':index_dtype.cases,'score_traversal':traversal.score_cases,
          'aggregation_traversal':traversal.aggregation_cases,'projected_layout':layouts.projected_cases,
          'attention_layout':layouts.attention_cases,'softmax_schedule':scheduling.softmax_cases,
          'aggregation_schedule':scheduling.aggregation_cases}


class AxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure(1);cls.native=Native()

    def graph(self,n=13,high=False,scale=1.):
        generator=torch.Generator().manual_seed(15)
        u=torch.randint(n,(n*3,),generator=generator);v=torch.randint(n,(n*3,),generator=generator)
        if high:u=torch.cat([u,torch.arange(n),torch.arange(n)]);v=torch.cat([v,torch.zeros(n,dtype=torch.long),torch.zeros(n,dtype=torch.long)])
        g=dgl.add_self_loop(dgl.remove_self_loop(dgl.graph((u,v),num_nodes=n+1)))
        g.ndata['feat']=torch.randn(n+1,7,generator=generator)*scale
        g.edata['payload']=torch.arange(g.num_edges(),dtype=torch.float32)
        g.create_formats_();return g

    def test_every_family_intermediates_repeatability(self):
        g=self.graph();model=model_for(7)
        with torch.no_grad():model.bias.copy_(torch.arange(32)*.01)
        with torch.inference_mode():
            for family,builder in FAMILIES.items():
                threads=4 if family.endswith('_schedule') else 1;torch.set_num_threads(threads)
                for value,case in builder(g,self.native,threads).items():
                    with self.subTest(family=family,value=value):
                        self.assertFalse(isinstance(case,dict),case)
                        check=validate(case,g,g.ndata['feat'],model)
                        self.assertEqual(check['status'],'passed')

    def test_irregular_high_degree_parallel_edges_numerical_extremes(self):
        with torch.inference_mode():
            for scale in [0.,1e-25,100.,1000.]:
                g=self.graph(n=65,high=True,scale=scale);model=model_for(7)
                for zlayout,elayout in [(0,0),(1,0),(0,1)]:
                    # Separate correctness coverage, never a layout performance grid.
                    case=Pipeline(g,native=self.native,zlayout=zlayout,elayout=elayout,threads=4,
                                  softmax_schedule='degree_balanced',aggregation_schedule='degree_balanced')
                    validate(case,g,g.ndata['feat'],model)

    def test_maps_restored_after_nontrivial_node_and_edge_permutation(self):
        g=self.graph(n=21,high=True);model=model_for(7)
        changed=dgl.reorder_graph(g,node_permute_algo='rcmk',edge_permute_algo='dst')
        changed.create_formats_();ids=changed.ndata[dgl.NID];eids=changed.edata[dgl.EID]
        self.assertTrue(torch.equal(changed.edata['payload'],g.edata['payload'][eids]))
        with torch.inference_mode():
            expected,attention=model(g,g.ndata['feat'],get_attention=True)
            case=Pipeline(changed,native=self.native,elayout=1,threads=4)
            output,details=case.forward(changed.ndata['feat'],model,details=True)
            restored=torch.empty_like(output);restored[ids]=output
            torch.testing.assert_close(restored,expected,rtol=RTOL,atol=ATOL)
            restored_a=torch.empty_like(attention);restored_a[eids]=details['intermediates']['attention']
            torch.testing.assert_close(restored_a,attention,rtol=RTOL,atol=ATOL)

    def test_physical_layouts_are_not_labels(self):
        g=self.graph();model=model_for(7)
        with torch.inference_mode():
            zcase=layouts.projected_cases(g,self.native,1)['hnf'];_,z=zcase.forward(g.ndata['feat'],model,details=True)
            self.assertEqual(z['tensors']['projected_physical']['logical_shape'],[4,g.num_nodes(),8])
            self.assertEqual(z['tensors']['projected_logical']['strides'],[8,g.num_nodes()*8,1])
            acase=layouts.attention_cases(g,self.native,1)['he'];_,a=acase.forward(g.ndata['feat'],model,details=True)
            self.assertEqual(a['tensors']['attention_physical']['logical_shape'],[4,g.num_edges()])
            self.assertEqual(a['tensors']['attention_logical']['strides'][:2],[1,g.num_edges()])

    def test_partition_ownership_and_actual_teams(self):
        g=self.graph(n=101,high=True);ptr=g.adj_tensors('csc')[0]
        a=row_partition(ptr,4,'static');b=row_partition(ptr,4,'degree_balanced')
        self.assertNotEqual(a['row_order_sha256'],b['row_order_sha256'])
        self.assertEqual(sum(b['edge_counts']),g.num_edges());self.assertEqual(self.native.lib.axes_team_size(4),4)

    def test_profiled_path_matches_uninstrumented(self):
        g=self.graph();model=model_for(7)
        with torch.inference_mode():
            for native in [None,self.native]:
                case=Pipeline(g,native=native,threads=1)
                normal=case.forward(g.ndata['feat'],model)
                profiled,detail=case.forward(g.ndata['feat'],model,profile=True)
                torch.testing.assert_close(normal,profiled,rtol=0,atol=0)
                self.assertEqual(len(detail['stages']),9)
                self.assertTrue(all(v>0 for v in detail['stages'].values()))

    def test_zero_degree_guard(self):
        g=dgl.graph(([0],[0]),num_nodes=2);g.ndata['feat']=torch.zeros(2,7);g.create_formats_();model=model_for(7)
        case=Pipeline(g,native=self.native,threads=1)
        with self.assertRaises(dgl.DGLError):case.forward(g.ndata['feat'],model)
        with self.assertRaises(dgl.DGLError):model(g,g.ndata['feat'])


if __name__=='__main__':unittest.main()
