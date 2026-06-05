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

files = [
    None,
    "NewResults\handpicked\BO_results_d_0_0.5.pkl",
    "NewResults\handpicked\BO_results_d_2_1.0.pkl",
    "NewResults\handpicked\BO_results_d_6_0.5.pkl",
    "NewResults\handpicked\BO_results_d_7_0.5.pkl",
    "NewResults\handpicked\BO_results_d_9_0.5.pkl",
    "NewResults\handpicked\GB_BO_results_0_0.5_4_72_10246.pkl",
    "NewResults\handpicked\GB_BO_results_1_0.5_4_72_8595.pkl",
    "NewResults\handpicked\GB_BO_results_4_0.5_6_72_16549.pkl",
    "NewResults\handpicked\GB_BO_results_11_0.5_8_72_526394.pkl",
]

pp_list = [
    0.05,
    0.040036870145840404,
    0.032059019421497734,
    0.0256708559516296,
    0.020555614525359374,
    0.01645964939039528,
    0.013179856905786339,
    0.010553604389554513,
    0.008450665770303305,
    0.0067667641618306355,
]

ler_results = []

for file in files:
    if file is None:
        code = cc.construct(gross_code)
        best_x = gross_code
        best_y = 0
    else:
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
