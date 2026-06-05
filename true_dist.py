import os
import pickle
from code_construction.code_construction import CodeConstructor, CSSCode
import codedistance

def get_nkd(code: CSSCode):
    res = codedistance.CSScodeDistance(
        code.hx,
        code.hz,
        method="MIPDist",  # exact distance finding algorithm
        params={'solverType': 'CP_SAT'},
        seed=101,
    )
    dist = res["d"]
    return code.n, code.k, dist


cc = CodeConstructor(method="bb", para_dict={"l": 12, "g": 6})
cc2 = CodeConstructor(method="gb", para_dict={"l":72})

gross_code = [
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
]

folder_path = "data/BO_results/lorenzo_results"

for file_name in [None] + os.listdir(folder_path):
    if file_name is None:
        file = "gross code"
        code = cc.construct(gross_code)
        best_x = gross_code
        best_y = 0
    else:
        file = os.path.join(folder_path, file_name)
        with open(file, "rb") as f:
            results = pickle.load(f)
            best_x = results["best_x"]
            best_y = results["best_y"]

        best_x = best_x.cpu()
            
        if "GB" in file:
            code = cc2.construct(best_x)
        else:
            code = cc.construct(best_x)

    n, k, d = get_nkd(code)

    print(f"from {file}, score: {best_y:.5f}")
    print(f"[[{n}, {k}, {d}]]")
    print("-------------------------------")

    with open("distance_res.txt", 'a') as f:
        f.write(f"\n[[{n}, {k}, {d}]], {best_y:.5f}\n")
