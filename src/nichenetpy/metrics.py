from sklearn.metrics import precision_recall_curve

'''
calculates the area under the curve using the trapezoid rule

the points should be specified in descending order of the x-values

Parameters
----------
x : list or tuple of float
    list of x-values on the curve
y : list or tuple of float
    list of y-values on the curve

Returns
-------
float
    the area under the curve

Raises
------
ValueError
    if x and y do not have matching length of at least 2

Examples
--------
>>> _auc_reverse(
    (1, 1, 1, 1, 1, 1, 0.85714286, 0.85714286, 0.57142857, 0.42857143, 0.42857143, 0.28571429, 0.14285714, 0),
    (0.5, 0.53846154, 0.58333333, 0.63636364, 0.7, 0.77777778, 0.75, 0.85714286, 0.8, 0.75, 1, 1, 1, 1)
)
0.8851473922902493
'''
def _auc_reverse(x:list[float], y:list[float]) -> float:
    if len(x) != len(y):
        raise ValueError('x and y should have the same length')
    if len(x) < 2:
        raise ValueError('x and y should have a length of at least 2')
    return sum((x[i-1] - x[i])*(y[i] + y[i-1]) for i in range(1, len(x))) / 2


'''
calculates the area under the precision-recall curve using the trapezoid rule

Parameters
----------
response : list or tuple of float
    vector indicating whether a target is a TRUE target of the possibly active ligand(s) or a FALSE
prediction : list or tuple of float
    vector which contains probability scores for each target gene (for one particular ligand)

Returns
-------
float
    the area under the precision-recall curve

Examples
--------
>>> calculate_aupr(
    (1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
    (0.5, 0.12, 0.47, 0.36, 0.0, 1.0, 0.78, 0.66, 0.24, 0.42, 0.95, 0.47, 0.39, 0.07)
)
0.8851473922902493
'''
def calculate_aupr(response, prediction):
    precision, recall, _ = precision_recall_curve(response, prediction)
    return _auc_reverse(recall, precision)