"""Report the seven independent axes spanning all four milestone families."""
import argparse
import json
from pathlib import Path
from analysis import analyze

TITLES={'index_dtype':'Sparse index dtype','score_traversal':'COO versus CSR additive score traversal',
 'aggregation_traversal':'CSC versus COO weighted aggregation','projected_layout':'Physical projected-feature NHF versus HNF',
 'attention_layout':'Physical score/attention EH versus HE','softmax_schedule':'Four-thread softmax row scheduling',
 'aggregation_schedule':'Four-thread aggregation row scheduling'}


def table(head,rows):
    return ['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+[
        '| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows]+['']


def ci(r):return f"{r['geomean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


def render(source,output):
    raw=json.loads(source.read_text());data=analyze(raw);output.parent.mkdir(parents=True,exist_ok=True)
    (output.parent/'ANALYSIS.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    first=next(v['rounds'][0] for v in raw['datasets'].values() if v['status']=='completed')
    env=first['environment'];opts=raw['options']
    lines=['# Milestone 5 — Independent DGL GAT CPU tuning axes','',
        '## 1. Research objective and scope','',
        'Evaluate four families independently: graph index dtype, stage-specific sparse traversal, physical intermediate layouts and sparse scheduling. '
        'There is no joint/factorial search, network architecture change, predictive cost tensor or model fit. '
        'All cases retain native dataset features/topology, four heads, eight output features per head, float32, identical weights, no dropout, no residual/activation and output bias enabled. '
        'The complete untouched DGL 1.1.3 GATConv is the oracle and an independently timed performance reference.','',
        'The seven contrasts below are one-parameter experiments. A slower custom family can still expose a useful internal trade-off; its gap from official DGL is shown separately and must not be attributed entirely to the named parameter.','',
        '## 2. Environment and reproducibility','']
    lines+=table(['Field','Recorded value'],[[k,env.get(k)] for k in ['cpu_model','logical_cpus','cpu_affinity','os','python','dgl','torch','torch_threads','torch_interop_threads','thread_environment','runtime_thread_controls','loaded_openmp_libraries','git_commit','cpu_quota','lock_sha256']])
    lines+=table(['Dataset','Nodes','Edges incl. self-loops','Native input width','Mean incoming degree','Max incoming degree'],
        [[ds,g['dataset_info']['nodes'],g['dataset_info']['edges_with_self_loops'],g['dataset_info']['original_features'],
          f"{g['dataset_info']['incoming_degree_summary']['mean']:.3f}",g['dataset_info']['incoming_degree_summary']['max']]
         for ds,g in data['datasets'].items() if 'dataset_info' in g])
    lines+=['One-thread groups contain A/B/C; four-thread groups contain the two D scheduling contrasts. '
        'Each axis has '+str(opts['rounds'])+' independent fresh-process rounds per dataset. Groups share a process only within a round; each parameter still has distinct processes across its rounds. '
        'No two measurement processes run concurrently. Both schedule values use exactly four PyTorch/OpenMP/MKL threads, inter-op one; the actual compiled OpenMP team is checked. '
        'Threads are an external fixed control, not a hidden tuning factor. OMP_WAIT_POLICY=PASSIVE and KMP_BLOCKTIME=0 are fixed before runtime import to avoid idle-team spinning; the loaded OpenMP libraries are recorded.','',
        f"Each block uses {opts['warmup']} warmups and {opts['repetitions']} timed forward repetitions. Case order and axis order are randomized from recorded seeds. "
        'Before and after each candidate, the same within-family reference is measured; reference/case speedup uses the mean of these bracketing median times. '
        'Each axis also has official-DGL blocks before and after its case groups. A reference row is normalized to one by definition, not a claim of zero timing noise.','',
        'Ordinary full-forward passes have no internal clock reads, profiler callbacks or stage dictionaries. Separate instrumented passes collect logical-stage timing; do not sum these to predict uninstrumented latency. '
        'Per-thread service clocks are enabled only for separate diagnostic calls. Per-round process peak RSS spans loading, validation and all axes in that process; it is not per-case peak memory.','',
        'The compiler is '+raw['compiler_initial_build']['compiler']+'. Flags: `'+ ' '.join(raw['compiler_initial_build']['flags'])+'`. '
        'No fast-math or accelerator build is used. The OpenMP C ABI library is loaded via stdlib ctypes, with no Torch extension headers or new Python runtime dependencies. '
        'Compilation and its cache status are recorded separately; binary objects live only in ignored results paths. New source hashes, compiler/binary digests and the historical base commit are retained.','',
        '## 3. Exact implementations and fixed references','']
    descriptions={
    'index_dtype':'Official g.long()/g.int(), all formats materialized, original orders, one thread; complete official GATConv. Check counts, row pointers, EIDs, actual index min/max and flattened EH/ND/NHF bounds before narrowing. Topology and data associations are verified through every materialized representation.',
    'score_traversal':'One DGL-primitive orchestrator uses gsddmm(add,l,r) on a sole COO or sole CSR view. Softmax and aggregation both stay CSC; original EIDs, NHF, threads and score arithmetic are unchanged. The degree guard executes on the canonical graph, not the CSR-only score view.',
    'aggregation_traversal':'Same orchestrator, COO score route and CSC destination softmax, varying only gspmm(mul,sum) on sole CSC or sole COO. The generic CPU routes differ in destination ownership versus edge-oriented atomic accumulation. Both are measured at ONE thread here; this is not a parallel thread-count comparison.',
    'projected_layout':'One common compiled family consumes logical [N,H,F] in NHF or physical contiguous [H,N,F]. The shared Torch projection initially produces NHF; HNF packing occurs each forward and is included. Native node-contribution and aggregation address functions read HNF directly, without conversion back to NHF. All loop organizations, edge layout and schedules are fixed.',
    'attention_layout':'Same compiled family, fixed NHF features and static row ownership, scores/attention physically [E,H] or contiguous [H,E]. Producers and all consumers use the corresponding eoff address function; no repack to EH occurs. Add and LeakyReLU remain separate passes. This is not an EID permutation or a transpose label erased by DGL.',
    'softmax_schedule':'Common compiled family at FOUR threads: contiguous destination-row ownership versus deterministic largest-degree-first greedy ownership, weighted by degree+1, with node-ID ties and sorted rows inside each thread. Whole rows remain intact; per-row head/neighbor math order is identical. Aggregation ownership stays static.',
    'aggregation_schedule':'Same four-thread family, only destination-row-to-thread ownership changes. Softmax ownership stays static. There are no atomics, row splits or merging partial outputs; each complete destination row has exactly one owner.'}
    for axis,desc in descriptions.items():lines += [f'**{axis}** — {desc}','']
    lines+=['The custom family is not asserted identical in overhead to stock DGL: it computes the two node contributions in one compiled node loop, uses output scratch for a three-pass stable softmax, and calls through ctypes. '
        'These choices stay fixed within layout/scheduling contrasts. Comparing a custom case directly to stock does not isolate layout or scheduling. Output bias is a separate Torch addition. '
        'No source/destination projected-feature dot product replaces additive GAT attention.','',
        '### Backend-route evidence','',
        'Every stage-specific DGL view records exactly one allowed/materialized representation. A successful gsddmm on a sole CSR view cannot silently use COO; a successful sole-COO gspmm cannot silently use CSC. '
        'The audited v1.1.3 dispatcher switches CSR/COO for generalized SDDMM and CSC/COO for generalized SpMM; CPU edge softmax receives CSC. '
        'Thus runtime view restrictions plus successful primitives and source dispatch identify different routes. No dynamic C++ symbol trace is claimed. '
        'Graph states are checked again after timing. Internal optimized-versus-generic subkernel selection is not established by these records.','',
        'The v1.1.3 CPU source has an edge-parallel COO score loop and a source-row CSR score loop. COO sum aggregation loops over edges with atomic updates; CSC is passed to the incoming-row CSR interface. '
        'See SOURCES.json for the independently retrieved, hash-verified source URLs and exact dispatch evidence. This is also the D1 edge-processing scheduling comparison; it is not added a second time as a nominal schedule axis.','',
        'This does not make a sole-format complete stock layer valid: COO-only still lacks CSC softmax, CSR-only cannot execute the stock incoming-degree guard, and CSC-only cannot execute score SDDMM. '
        'The preserved historical pilot and regression checks retain these failures. The new orchestration calls each primitive on a compatible view and leaves the degree check on the canonical graph; it does not silently broaden any stage mask.','',
        '### Physical addresses and copies','',
        'NHF strides are (H*F,F,1); HNF physical strides are contiguous [H,N,F], exposing logical [N,H,F] strides (F,N*F,1). '
        'EH logical strides are (H,1); HE storage exposes logical [E,H] strides (1,E). Tensor metadata records both physical and logical shapes/strides. '
        'Default NHF packing is an identity. HNF packing reads/writes Z each forward; HE scores/attention are produced directly in that storage. All real-data output nodes/EIDs remain canonical, so no per-forward node restoration or edge remapping is required.','',
        '## 4. Numerical validation','',
        'The complete layer and accessible intermediates are checked at rtol=1e-4, atol=1e-5. Reference intermediate calculations use the official layer parameters and matching DGL primitives, and attention/final output are cross-checked with untouched GATConv(get_attention=True). '
        'We compare Z, node l/r, raw additive logits, LeakyReLU logits, normalized attention, aggregated values and final output wherever exposed. Index dtype uses the untouched complete layer and exposes output/attention rather than inventing its logical-stage profile. '
        'Repeated forwards must be bitwise repeatable within the tested process/configuration. All real inputs are finite. Tests additionally cover zero/tiny/scaled finite features, high-degree destinations, parallel edges, self-loops, initially isolated nodes, nondivisible sizes, nonzero bias and nontrivial restored node/EID mappings. '
        'Zero-in-degree guards remain enabled. Stable softmax subtracts the full destination/head maximum and normalizes the full neighborhood, never independent chunks.','']
    rows=[]
    for dataset,g in data['datasets'].items():
        if g.get('status')=='skipped_memory':continue
        for axis,a in g['axes'].items():
            for value,c in a['cases'].items():rows.append([dataset,axis,value,c['status'],c.get('output_max_absolute_error','—'),c.get('max_scaled_intermediate_error','—'),c.get('bitwise_repeatable','—')])
    lines+=table(['Dataset','Axis','Value','Status','Max output abs error','Max scaled intermediate error','Repeatable'],rows)
    section=5
    for axis,title in TITLES.items():
        lines += [f'## {section}. {title}',''];section+=1
        decision=data['axes'].get(axis,{'classification':'C','reason':'No valid records'})
        lines += [f"**Recommendation {decision['classification']}** — {decision['reason']}",'']
        rows=[]
        for dataset,g in data['datasets'].items():
            if axis not in g.get('axes',{}):continue
            a=g['axes'][axis]
            for value,c in a['cases'].items():
                if c['status']!='measured':rows.append([dataset,value,'excluded','—','—','—','—']);continue
                rows.append([dataset,value,f"{c['median_round_median_ms']:.5f}",f"{c['mean_round_mean_ms']:.5f}",f"{c['between_round_sd_ms']:.5f}",ci(c['speedup_vs_reference']),ci(c['official_over_case_speedup'])])
        lines+=table(['Dataset','Value','Median of round medians ms','Mean round mean ms','Between-round SD ms','Within-family speedup [95% CI]','Official/case speedup [95% CI]'],rows)
        lines+=['Speedup above one favors the candidate; official/case below one means the custom/primitive family is slower than official DGL. '
            'Intervals are t intervals of independent-round log ratios, unadjusted for multiple comparisons. Rankings use within-family normalized times, not absolute latencies of unrelated families.','']
        winners=decision.get('best_observed',{})
        lines += [f'Best observed within-family values: **{winners}**. Credible preference directions: **'+str(decision.get('credible_preference',{}))+'**. '
            'A changed point-estimate winner is not automatically a statistically credible crossover.','']
        if axis=='index_dtype':
            rows=[]
            for dataset,g in data['datasets'].items():
                if axis not in g.get('axes',{}):continue
                for value,c in g['axes'][axis]['cases'].items():
                    if c['status']=='measured':rows.append([dataset,value,c['representative_implementation']['graph']['idtype'],c['representative_implementation']['graph']['unique_sparse_storage_payload_bytes'],f"{c['one_time_parameter_preparation_ms']['mean']:.4f}"])
            lines+=table(['Dataset','Value','Actual dtype','Unique sparse tensor payload bytes','Cast+materialization ms'],rows)
            lines+=['Payload deduplicates exposed index-array storages inside one graph; features, hidden graph metadata, allocator overhead, other resident views and whole-process peak are not included. '
                'A smaller payload does not establish faster execution or that Reddit fits. If one width dominates across this envelope, use it as a legality-checked default here, not a universal rule for larger graphs or other CPUs.','']
        if axis in ['projected_layout','attention_layout']:
            rows=[]
            for dataset,g in data['datasets'].items():
                if axis not in g.get('axes',{}):continue
                for value,c in g['axes'][axis]['cases'].items():
                    if c['status']!='measured':continue
                    label='projected' if axis=='projected_layout' else 'attention'
                    logical=c['representative_tensors'][label+'_logical'];physical=c['representative_tensors'][label+'_physical']
                    pack=c['profile_stages'].get('projected_packing',{}).get('mean_median_ms') if axis=='projected_layout' else 0
                    rows.append([dataset,value,logical['logical_shape'],logical['strides'],physical['logical_shape'],pack,c['representative_intermediate_payload_bytes']])
            lines+=table(['Dataset','Value','Logical shape','Actual logical strides','Physical shape','Profiled packing ms','Detailed live intermediate payload bytes'],rows)
            lines+=['The memory column is live exposed tensor payload in detailed validation, not allocator peak. HNF packing temporarily overlaps the original projection buffer, so its packing peak can exceed the end-of-validation payload. '
                'Profiled packing includes stage-call timing overhead; the identity path performs no copy.','']
        if axis.endswith('_schedule'):
            rows=[];part_key='softmax_partition' if axis=='softmax_schedule' else 'aggregation_partition'
            for dataset,g in data['datasets'].items():
                if axis not in g.get('axes',{}):continue
                for value,c in g['axes'][axis]['cases'].items():
                    if c['status']!='measured':continue
                    part=c['representative_implementation'][part_key]
                    rows.append([dataset,value,part['row_counts'],part['edge_counts'],f"{c['one_time_parameter_preparation_ms']['mean']:.4f}",c['representative_implementation']['actual_openmp_team']])
            lines+=table(['Dataset','Schedule','Rows per thread','Edges per thread','Partition preparation ms','Actual threads'],rows)
            lines+=['Greedy assignment balances a degree+1 work proxy, not measured cycles. Row migration can hurt locality; a single heavy row is unsplittable in both alternatives. '
                'Per-thread service times in RESULTS.json are separate single diagnostic calls per round, exclude the final team barrier and are not standalone performance-confidence estimates. '
                'Each stage has one implicit team-completion barrier. Softmax uses thread-local max/sum scalars and alpha as exponent scratch; aggregation has no private full-output buffers or cross-thread reduction. '
                'The two cached ownership maps each contain N int64 row IDs and threads+1 int64 boundaries, or 2×8×(N+threads+1) payload bytes per case; this excludes allocator overhead and shared graph arrays. '
                'No 1-vs-4-thread scheduling comparison is made.','']
    lines += ['## 12. Stage profiles and end-to-end interpretation','',
        'All instrumented passes are separate from ordinary samples. Stage means below are averages of round profile medians; they include stage-call clock overhead and cannot be summed into an uninstrumented prediction. '
        'Official dtype-only stage profiles are unavailable and explicitly marked so in raw data.','']
    for axis in TITLES:
        rows=[]
        for dataset,g in data['datasets'].items():
            if axis not in g.get('axes',{}):continue
            for value,c in g['axes'][axis]['cases'].items():
                if c['status']!='measured':continue
                s=c['profile_stages']
                if not s:continue
                rows.append([dataset,value,*[f"{s[k]['mean_median_ms']:.5f}" for k in ['projection','projected_packing','node_contributions','edge_addition','leaky_relu','edge_softmax','aggregation','output_bias']]])
        if rows:lines += ['**'+axis+'**','']+table(['Dataset','Value','Projection','Z pack','Node l/r','Edge add','Leaky','Softmax','Aggregate','Bias'],rows)
    lines += ['![Independent axis effects](figures/axis-effects.svg)','',
        'Hypotheses: dtype benefits depend on index traffic relative to dense projection; CSR/COO differ in endpoint reuse and ownership; HNF may help one consumer while packing or another consumer loses locality; HE can favor head-wise softmax while worsening edge-major writes; degree balancing can reduce row-load skew while reducing contiguous traversal locality. '
        'These explanations are hypotheses, not measured hardware-counter causes. A stage improvement need not survive full-forward overheads or dominant dense projection.','',
        '## 13. Preprocessing, conversion and memory','',
        'Recorded separately: dataset load and shared self-loop/all-format preparation; fresh-COO-to-COO/CSR/CSC view materialization diagnostics; index casting; constructor view wrapping; scheduling partition construction; compiler first build/cache metadata. '
        'Graph-view diagnostics construct source COO tensors before their conversion timer, so they isolate restriction/materialization rather than claiming a complete load cost. '
        'No EID remapping is performed every forward: all views communicate in canonical EID order. HNF packing is per-forward and always included; HE is produced directly. '
        'Per-case stage tensors record strides, storage size and contiguity. Compilation is shared by all custom cases; it cannot be assigned only to the slower layout.','']
    rows=[]
    cold=[]
    for dataset,record in raw['datasets'].items():
        workers=[w for w in record.get('rounds',[]) if w['threads']==1]
        for fmt in ['coo','csr','csc']:
            if workers:cold.append([dataset,fmt,f"{sum(w['cold_sparse_format_construction'][fmt]['restriction_and_materialization_ms'] for w in workers)/len(workers):.5f}"])
    lines+=table(['Dataset','Cold sole-format view from fresh COO','Mean construction ms (one-thread workers)'],cold)
    lines += [f"The primary native library's uncached build took {raw['compiler_initial_build']['build_seconds']:.3f} seconds; per-worker cache metadata distinguishes reuse from compilation. "
        'Compilation is excluded from forward samples. No compiled objects are versioned.','']
    for dataset,g in data['datasets'].items():
        if 'axes' not in g:continue
        for axis,a in g['axes'].items():
            for value,c in a['cases'].items():
                if c['status']=='measured':rows.append([dataset,axis,value,f"{c['one_time_parameter_preparation_ms']['mean']:.5f}"])
    lines+=table(['Dataset','Axis','Value','Mean one-time parameter preparation ms'],rows)
    lines+=['Layout-specific preparation is zero: required HNF packing is a recurring cost, already timed. Shared graph setup is not zero; it is stored separately. '
        'For deployment, incremental preprocessing only amortizes if the measured steady-state saving remains positive. Peak RSS cannot identify which value caused a peak because the process also holds other case views and diagnostic intermediates.','',
        '## 14. Cross-dataset recommendations, exclusions and uncertainty','']
    admitted=[a for a,d in data['axes'].items() if d['classification']=='A']
    if not admitted:
        lines+=['**Research finding: no new contrast demonstrated a credible, practically useful cross-dataset ranking reversal in this run.** '
            'That is evidence about this CPU and three citation workloads, not proof that these implementation families cannot supply useful tensor dimensions. '
            'Do not promote a pair merely because both values execute or because noisy point-estimate winners differ.','']
    lines+=['Best observed means the **bracketing-control-normalized** point estimate. It can differ from the smallest unpaired median in the table because the blocks occur at different times. '
        'In particular, an inconclusive point-estimate winner must not be described as a dataset preference established by the experiment.','']
    rows=[]
    for axis,d in data['axes'].items():rows.append([axis,d['classification'],d.get('best_observed','reuse traversal results'),d['reason']])
    lines+=table(['Parameter','Class','Best observed per dataset','Reason'],rows)
    if data['axes'].get('aggregation_traversal',{}).get('classification')=='E':
        advantages=[]
        for dataset,g in data['datasets'].items():
            if 'aggregation_traversal' in g.get('axes',{}):
                p=g['axes']['aggregation_traversal']
                if p['best_observed']=='csc':advantages.append(f"{dataset} {1/p['cases']['coo']['speedup_vs_reference']['geomean']:.2f}×")
        if advantages:lines += ['**Measured aggregation default: CSC.** Its paired full-forward speedups over COO were '+', '.join(advantages)+
            '. Collapse COO aggregation from the current one-thread search space; this does not prove dominance for a different CPU, dtype, thread budget or kernel family.','']
    lines += ['A=retain meaningful axis; B=real but inconclusive; C=required implementation incomplete; D=conditional useful choice; E=collapse/exclude in the tested envelope. '
        'The automatic rule is declared before full measurement: opposite credible directions with ≥3% effects across the dataset envelope admit A; a consistent ≥3% interval-supported preference across all datasets collapses that value pair as E for this envelope; useful ≥3% effects without a qualifying reversal are D; small/uncertain effects are B. '
        'This does not exclude an untested family or extrapolate a dominant choice beyond the tested graph shapes, CPU, layouts and resource budgets.','']
    lines+=['### Interpretation and next independent checks','',
        '**Index width:** both widths preserve topology, EIDs and the complete official layer. The exposed sparse payload is halved by int32; this is a memory fact, not a latency result. '
        'Keep width as a legality/memory-conditioned candidate while the latency intervals remain inconclusive. Test a larger safely resident workload before selecting int32 as a speed default.','',
        '**Stage traversal:** sole-format views establish real alternatives without forcing the complete layer onto an unsupported format. '
        'Use the CSC/COO aggregation comparison to decide the default only within this stock CPU, one-thread envelope. '
        'The score route still deserves separate, stage-focused measurement: a small score-stage change can be hidden by projection, softmax and aggregation. '
        'Stage profiles are descriptive and have no separately claimed stage-level significance test.','',
        '**Projected layout:** this tests a real HNF consumer but a shared NHF-producing projection, so the price of HNF includes packing on every call. '
        'A head-major-friendly stage is insufficient if packing or aggregation offsets its gain. '
        'Prefer the observed default in the present family; a direct HNF producer would be an additional implementation study, not an assumed free transpose or a result already established here.','',
        '**Edge layout:** EH and HE are produced and consumed directly. HE changes edge-add writes, stable softmax reads and aggregation reads together because one shared intermediate crosses those boundaries. '
        'Retain a conditional layout candidate when intervals are inconclusive; do not infer a universal layout from a single profiled stage.','',
        '**Softmax and aggregation schedules:** static and greedy ownership are real four-thread alternatives. Degree+1 balance improves a proxy, while noncontiguous rows and team/scheduling overhead can offset that benefit. '
        'Their best observed full-forward value must be distinguished from evidence of improved thread-load balance. '
        'Next test one scheduling axis at a time on a safely resident real graph with a broader degree tail; do not change head count or native feature width to fit. '
        'Whole-row ownership cannot resolve a single dominant hub; correct split-row reductions remain deferred implementation work.','',
        '**Shortlist:** retain A contrasts if any appear in the decision table; retain B contrasts as unproven research candidates and D contrasts only within their named consumer/resource family. '
        'Collapse E value pairs in the measured envelope, retaining their raw evidence and legal alternatives for future architectures. '
        'Do not duplicate D1 traversal as another independent axis, combine the layouts/schedules into a factorial search, or construct a cost tensor from unsupported combinations. '
        'The custom consumers currently require int64 sparse arrays; official index width cannot simply be crossed with them. All scheduling cases fix NHF/EH/CSC; all layout cases fix static ownership.','']
    lines+=['The next small independent studies should first revisit **index width and additive-score traversal** on a larger safely resident native-feature workload, then **each reduction schedule separately** on a broader-degree graph. '
        'Keep physical layouts conditional on matching producers/consumers and include every required pack. Predeclare fresh cohorts and sample budgets; do not add repetitions to this completed study until a preferred value becomes significant. '
        'No new automatic cost-tensor dimensions are justified solely by these inconclusive comparisons. Existing node/edge-ordering recommendations are preserved and were not remeasured here.','']
    drifts=[c['control_drift_ratio'] for g in raw['datasets'].values() for w in g.get('rounds',[]) for a in w['axes'].values() for c in a['cases'].values() if c['status']=='measured']
    lines += [f"**Observed drift:** {sum(abs(v-1)>.1 for v in drifts)} of {len(drifts)} before/after control brackets changed by more than 10%; after/before ratios ranged from {min(drifts):.3f} to {max(drifts):.3f}. "
        'All samples and all declared rounds are retained. Wide intervals and unstable point rankings remain limitations; no new rounds were selected to obtain significance. '
        'Bracketing reduces sensitivity to drift, but does not remove arbitrary within-block VM interference.','']
    for dataset,g in raw['datasets'].items():
        if g['status']=='skipped_memory':lines += [f"**{dataset}: skipped_memory** — {g['reason']}. Estimate {g['estimate_bytes']/2**30:.2f} GiB versus budget {g['budget_bytes']/2**30:.2f} GiB. No Reddit measurements were substituted from another shape or prior one-head work.",'']
    lines += ['Unsupported/incorrect cases are retained with exact errors and excluded from ranks. A completed worker is not proof every case passed; inspect the per-axis case statuses. '
        'All workload dimensions remain fixed. No high-degree row splitting, custom backward or arbitrary-stride kernel API is provided. Custom C ABI functions are inference-only for validated NHF/HNF and EH/HE layouts with int64 sparse arrays; int32 is studied using official DGL only. '
        'Citations graphs provide a narrow low-average-degree envelope; degree tails are recorded, but Reddit scheduling/layout behavior is unknown. '
        'Intervals are based on six independent process rounds, not the hundreds of intra-process repetitions. They are exploratory and unadjusted; there are no added rounds selected to achieve significance. '
        'One official-control pair spans each axis rather than every case; its gap estimates are more drift-sensitive than the within-family bracketed comparisons. '
        'VM placement, CPU frequency and shared-host interference remain uncontrolled. The observed base commit and all new source hashes identify the measured code.','',
        '## 15. Reproduction and validation','',
        'See README.md for the compiler build and Docker extension. Dependencies remain in the original frozen uv environment; plots are rendered separately with host Matplotlib 3.10.8.','',
        '```sh','export DGLBACKEND=pytorch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'export UV_CACHE_DIR=/workspace/.cache/uv',
        'uv run --frozen python benchmarks/tuning_axes/tests.py',
        'python benchmarks/tuning_axes/verify_results.py',
        'uv run --frozen python benchmarks/tuning_axes/run.py --datasets cora citeseer pubmed reddit \\',
        '  --rounds 6 --warmup 5 --repetitions 20 --seed 0 --order-seed 20261105',
        'uv run --frozen python benchmarks/tuning_axes/report.py',
        'python benchmarks/tuning_axes/plots.py','```','',
        'RESULTS.json retains full raw measurements, profiles, errors, states, thread assignments, bounds and provenance. ANALYSIS.json retains round-level comparisons and decisions. '
        'VALIDATION.json records native/Docker/regression checks and the offline raw-data audit. Measured implementation/driver hashes are verified against retained source; report/audit tools are untimed postprocessing. '
        'Historical baseline, joint-tuning and parameter-evaluation source/results are not modified. No predictive model, cost tensor, factorization or new joint study is implemented.']
    output.write_text('\n'.join(lines)+'\n');print(f'Wrote {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('benchmarks/tuning_axes/RESULTS.json'))
    p.add_argument('--output',type=Path,default=Path('benchmarks/tuning_axes/REPORT.md'))
    a=p.parse_args();render(a.input,a.output)
