import numpy as np

print(np.average(np.array([1,2,3])))
# y_true = np.array(['a', 'b', 'c', 'a'])
# y_pred = np.array(['a', 'c', 'c', 'b'])

# tp = np.sum(np.logical_and((y_true == 'a'), (y_true == y_pred)))
# tn = np.sum(np.logical_and((y_true != 'a'), (y_true == y_pred)))

# fp = np.sum(np.logical_and((y_true != 'a'), (y_true != y_pred)))
# fn = np.sum(np.logical_and((y_true == 'a'), (y_true != y_pred)))



# '''
# recall =  TP / (TP + FN)

# '''