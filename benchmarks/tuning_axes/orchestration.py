"""Same fixed GAT pipeline, stock primitives or a common layout-aware CPU family."""
from time import perf_counter_ns
from common import *


class Official:
    def __init__(self,g):self.g=g;self.info={'implementation':'untouched official GATConv','graph':view_info(g)}
    def forward(self,x,layer,details=False,profile=False):
        if details:
            y,a=layer(self.g,x,get_attention=True)
            return y,{'intermediates':{'output':y,'attention':a},'tensors':{'input':tensor_info(x)}}
        y=layer(self.g,x)
        if profile:return y,{'stages':{},'note':'Official whole-layer timing; no substitute orchestration claimed as its stage profile'}
        return y


class Pipeline:
    def __init__(self,g,score='coo',aggregation='csc',native=None,zlayout=0,elayout=0,
                 softmax_schedule='static',aggregation_schedule='static',threads=1):
        self.g=g;self.n=g.num_nodes();self.e=g.num_edges();self.native=native
        self.zlayout=zlayout;self.elayout=elayout;self.threads=threads
        begin=perf_counter_ns()
        self.score_g=g.formats(score);self.score_g.create_formats_()
        self.softmax_g=g.formats('csc');self.softmax_g.create_formats_()
        self.aggregation_g=g.formats(aggregation);self.aggregation_g.create_formats_()
        view_build_ms=(perf_counter_ns()-begin)/1e6
        for view in [self.score_g,self.softmax_g,self.aggregation_g]:validate_view(view,g)
        self.ptr,self.src,self.eids=self.softmax_g.adj_tensors('csc')
        self.ptr=self.ptr.long();self.src=self.src.long()
        self.eids=self.eids.long() if self.eids.numel() else torch.arange(self.e)
        self.u,self.v=g.edges(order='eid');self.u=self.u.long();self.v=self.v.long()
        self.softmax_partition=row_partition(self.ptr,threads,softmax_schedule)
        self.aggregation_partition=row_partition(self.ptr,threads,aggregation_schedule)
        self.info={'view_construction_ms':view_build_ms,'implementation':'compiled common layout-aware family' if native else 'DGL primitive orchestration',
            'score_primitive':'axes_add COO additive l+r' if native else f'dgl.ops.gsddmm(add), sole {score.upper()} view',
            'softmax_primitive':'axes_softmax stable CSC' if native else 'dgl.ops.edge_softmax(destination), CSC',
            'aggregation_primitive':'axes_aggregate CSC row ownership' if native else f'dgl.ops.gspmm(mul,sum), sole {aggregation.upper()} view',
            'views':{'score':view_info(self.score_g),'softmax':view_info(self.softmax_g),'aggregation':view_info(self.aggregation_g)},
            'canonical_eid_remapping_per_forward':False,'node_order':'original','edge_order':'original canonical EIDs',
            'zlayout':'HNF physical / NHF logical' if zlayout else 'NHF contiguous',
            'edge_layout':'HE physical / EH logical' if elayout else 'EH contiguous',
            'threads':threads,'softmax_partition':self.partition_info(self.softmax_partition),
            'aggregation_partition':self.partition_info(self.aggregation_partition)}
        if native:
            assert g.idtype==torch.int64
            self.info['actual_openmp_team']=native.lib.axes_team_size(threads)
            assert self.info['actual_openmp_team']==threads

    @staticmethod
    def partition_info(p):return {k:v for k,v in p.items() if k not in ['order','parts']}

    def guard(self):
        if (self.g.in_degrees()==0).any():raise dgl.DGLError('There are 0-in-degree nodes in the graph')

    def project(self,x,layer):return layer.fc(x).view(-1,H,F)
    def pack(self,z):return z.permute(1,0,2).contiguous() if self.zlayout else z
    def node_terms(self,z,layer):
        if self.native:return self.native.node(z,layer.attn_l,layer.attn_r,self.n,H,F,self.zlayout,self.threads)
        return (z*layer.attn_l).sum(-1).unsqueeze(-1),(z*layer.attn_r).sum(-1).unsqueeze(-1)
    def add(self,l,r):
        return self.native.add(l,r,self.u,self.v,self.e,H,self.elayout,self.threads) if self.native else dgl.ops.gsddmm(self.score_g,'add',l,r)
    def activate(self,q,layer):
        return self.native.leaky(q,layer.leaky_relu.negative_slope,self.threads) if self.native else layer.leaky_relu(q)
    def normalize(self,s):
        return self.native.softmax(self.ptr,self.eids,s,self.n,self.e,H,self.elayout,self.softmax_partition,self.threads) if self.native else dgl.ops.edge_softmax(self.softmax_g,s)
    def reduce(self,z,a):
        return self.native.aggregate(self.ptr,self.src,self.eids,z,a,self.n,self.e,H,F,self.zlayout,self.elayout,self.aggregation_partition,self.threads) if self.native else dgl.ops.gspmm(self.aggregation_g,'mul','sum',z,a)
    def bias(self,agg,layer):return agg+layer.bias.view(1,H,F)

    def profile_forward(self,x,layer):
        # Separate diagnostic path: ordinary forward below has NO clock reads,
        # stage dictionaries, profiling callbacks or profiler instrumentation.
        stages={}
        def timed(name,function):
            start=perf_counter_ns();value=function();stages[name]=(perf_counter_ns()-start)/1e6;return value
        timed('degree_guard',self.guard)
        z=timed('projection',lambda:self.project(x,layer))
        z=timed('projected_packing',lambda:self.pack(z))
        l,r=timed('node_contributions',lambda:self.node_terms(z,layer))
        q=timed('edge_addition',lambda:self.add(l,r))
        s=timed('leaky_relu',lambda:self.activate(q,layer))
        a=timed('edge_softmax',lambda:self.normalize(s))
        agg=timed('aggregation',lambda:self.reduce(z,a))
        y=timed('output_bias',lambda:self.bias(agg,layer))
        return y,{'stages':stages}

    def forward(self,x,layer,details=False,profile=False):
        if profile:return self.profile_forward(x,layer)
        self.guard()
        z=self.project(x,layer)
        # EXPERIMENTAL CHANGE: HNF packing is real, per-forward and included.
        z=self.pack(z)
        l,r=self.node_terms(z,layer)
        q=self.add(l,r)
        s=self.activate(q,layer)
        a=self.normalize(s)
        agg=self.reduce(z,a)
        y=self.bias(agg,layer)
        if details:
            logical_z=z.permute(1,0,2) if self.zlayout else z
            def edge(t):
                if self.native:return (t.transpose(0,1) if self.elayout else t).unsqueeze(-1)
                return t
            intermediates={'projected':logical_z,'source_contribution':l,'destination_contribution':r,
                'raw_logits':edge(q),'logits':edge(s),'attention':edge(a),'aggregated':agg,'output':y}
            tensors={'projected_logical':tensor_info(logical_z),'projected_physical':tensor_info(z),
                     'logits_logical':tensor_info(edge(s)),'logits_physical':tensor_info(s),
                     'attention_logical':tensor_info(edge(a)),'attention_physical':tensor_info(a)}
            unique={}
            for t in [z,l,r,q,s,a,agg,y]:unique[t.untyped_storage().data_ptr()]=t.untyped_storage().nbytes()
            return y,{'intermediates':intermediates,'tensors':tensors,'intermediate_payload_bytes':sum(unique.values()),
                      'memory_note':'Live exposed tensor payload at end of detailed validation; not peak allocator/RSS; original projection temporary for HNF not counted here'}
        return y

    def diagnostics(self,x,layer):
        if not self.native:return {'note':'Compiled per-thread service diagnostics do not apply to stock DGL primitives'}
        _,d=self.forward(x,layer,details=True)
        z=d['intermediates']['projected'];z=z.permute(1,0,2).contiguous() if self.zlayout else z.contiguous()
        s=d['intermediates']['logits'].squeeze(-1);s=s.transpose(0,1).contiguous() if self.elayout else s.contiguous()
        a,t1=self.native.softmax(self.ptr,self.eids,s,self.n,self.e,H,self.elayout,self.softmax_partition,self.threads,True)
        _,t2=self.native.aggregate(self.ptr,self.src,self.eids,z,a,self.n,self.e,H,F,self.zlayout,self.elayout,self.aggregation_partition,self.threads,True)
        return {'softmax_thread_service_seconds':t1.tolist(),'aggregation_thread_service_seconds':t2.tolist(),
                'note':'Separate diagnostic calls, excluding final team barrier; never included in ordinary forward timing'}
