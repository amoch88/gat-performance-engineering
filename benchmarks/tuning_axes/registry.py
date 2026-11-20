"""Seven independent axes across the four requested families; no factorial grid."""
import index_dtype,traversal,layouts,scheduling

AXES={
 'index_dtype':{'family':'A','threads':1,'reference':'int64','builder':index_dtype.cases},
 'score_traversal':{'family':'B','threads':1,'reference':'coo','builder':traversal.score_cases},
 'aggregation_traversal':{'family':'B','threads':1,'reference':'csc','builder':traversal.aggregation_cases},
 'projected_layout':{'family':'C','threads':1,'reference':'nhf','builder':layouts.projected_cases},
 'attention_layout':{'family':'C','threads':1,'reference':'eh','builder':layouts.attention_cases},
 'softmax_schedule':{'family':'D','threads':4,'reference':'static','builder':scheduling.softmax_cases},
 'aggregation_schedule':{'family':'D','threads':4,'reference':'static','builder':scheduling.aggregation_cases},
}
