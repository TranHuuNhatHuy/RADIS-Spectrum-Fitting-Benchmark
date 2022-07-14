# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json
import ast

# Get the result data from the file
with open("method_comparison.txt", "r") as f:
    result = json.load(f)
    f.close()

methods = [
    "leastsq",
    "least_squares",
    "differential_evolutions",
    "brute",
    #"basinhopping",
    #"ampgo",
    "nelder",
    "lbfgsb",
    "powell",
    "cg",
    "cobyla",
    "bfgs",
    "tnc",
    "trust-constr",
    "slsqp",
    "shgo",
    #"dual_annealing"
]

fileName = result.pop("fileName")
pipeline = result.pop("fitting-pipeline")

last_residuals = []
loops = []
time = []
for meth in methods:
    last_residuals.append(result[meth]["last_residual"])
    loops.append(result[meth]["loops"])
    time.append(result[meth]["time"])

# We only care about last_residual and loops
plt.scatter(
    x = loops,
    y = last_residuals,
    c = time,
    cmap = 'jet'
)

plt.title(fileName, fontsize = 20)
plt.xlabel("Loops", fontsize = 15)
plt.ylabel("Last residual", fontsize = 15)
plt.text(
    0.5,
    0.05,
    ast.literal_eval(str(pipeline)),
    ha = "center"
)
plt.colorbar(label = "Time (s)")

for i in range(len(methods)):
    plt.annotate(
        methods[i],
        (
            loops[i],
            last_residuals[i]
        )
    )

plt.show()

#plt.savefig("./result_scatter.png")