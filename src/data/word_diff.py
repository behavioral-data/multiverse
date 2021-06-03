from diff_match_patch import diff_match_patch

class dmp_with_pprint(diff_match_patch):
    def __init__(self, *args, **kwargs):
        self.pp_removed_start_token = kwargs.pop("pp_removed_start_token","<REMOVED>")
        self.pp_removed_end_token = kwargs.pop("pp_removed_end_token","</REMOVED>")
        self.pp_inserted_start_token = kwargs.pop("pp_inserted_start_token","<INSERTED>")
        self.pp_inserted_end_token = kwargs.pop("pp_inserted_end_token","</INSERTED>")
        
        super().__init__(*args, **kwargs)
    
    def pprint(self,diffs):
        pretty_diff = []
        for (op, text) in diffs:
            if op == self.DIFF_INSERT:
                pretty_diff.append(f"{self.pp_inserted_start_token}{text}{self.pp_inserted_end_token}")
            elif op == self.DIFF_DELETE:
                pretty_diff.append(f"{self.pp_removed_start_token}{text}{self.pp_removed_end_token}")
            elif op == self.DIFF_EQUAL:
                pretty_diff.append(text)
        return "".join(pretty_diff)


def compute_word_diff(a,b,semantic_cleanup=True):
    dmp = dmp_with_pprint()
    diff = dmp.diff_main(a,b)
    if semantic_cleanup:
        dmp.diff_cleanupSemantic(diff)
    return dmp.pprint(diff)


a = """model = xgb.XGBRegressor(learning_rate=0.1, gamma= 0)
model.fit(xtrain,ytrain)"""

b = """model = xgb.XGBRegressor(learning_rate=0.05, gamma= 0)
model.fit(xtrain,ytrain)"""
