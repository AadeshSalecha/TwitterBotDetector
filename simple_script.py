import numpy as np
from numpy.core.defchararray import array
from tensorflow.keras.models import load_model
import pickle as pkl

# model = load_model('./Models/warproxxx.h5')
x = np.array ([[205, 75, 635, 4487, 0, 0, 1, 0, 1], [22, 1, 45, 11, 0, 0, 1, 0, 1], [1105, 1745, 386, 1245, 3, 0, 1, 0, 1], [399, 673, 423, 571, 0, 0, 1, 0, 1], [29, 27, 158, 1105, 0, 0, 1, 0, 1], [49693, 13591, 214, 7373, 206, 0, 1, 0, 1], [52, 7, 146, 374, 0, 0, 1, 0, 1], [3591, 328, 1011, 5012, 0, 0, 1, 0, 1], [62, 12, 88, 507, 0, 0, 1, 0, 1], [2984, 223, 607, 2778, 1, 0, 1, 0, 1], [43, 10, 344, 598, 0, 0, 1, 0, 1], [29, 27, 158, 1105, 0, 0, 1, 0, 1]])

x = np.array([[35, 4, 38, 0, 0, 0, 1, 0, 1], [43, 5, 46, 0, 0, 0, 1, 0, 1], [38, 2, 36, 0, 0, 0, 1, 0, 1], [43, 2, 36, 0, 0, 0, 1, 0, 1], [44, 3, 39, 0, 0, 0, 1, 0, 1], [43, 3, 45, 0, 0, 0, 1, 0, 1], [36, 7, 44, 0, 0, 0, 1, 0, 1], [35, 8, 40, 0, 0, 0, 1, 0, 1], [41, 2, 37, 0, 0, 0, 1, 0, 1], ])

model = None
with open('./Models/adaboost_model', 'rb') as clf:
  model = pkl.load(clf)

preds = model.predict(x)
print(preds)

# Sample: "id": 832047828167950336, "screen_name": "aarjavjain20039", "name": "Aarjav jain", "statuses_count": 22, "favourites_count": 11, "followers_count": 1, "friends_count": 45, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Feb 16 02:03:55 +0000 2017", "location": "saharanpur"}"

# I also tried this for x
# x = np.array ([[22, 1, 45, 11, 0, 1, 1, 0, 1, 0]])

# None of the below statements worked for me
# print(model.predict( x ))
# print(model.predict( (x) ))
# print(model.predict( [x] ))
