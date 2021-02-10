import pickle as pkl
import numpy as np
import json

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.data import Graph

from tensorflow.keras.models import load_model

def main():
  # model = load_model('./Models/model0208/content/model0208/')

  # model.summary()

  print("Conversion started")
  # statuses_count followers_count friends_count favourites_count bg_image_false bg_image_true verified_false verified_true protected_false protected_true
  # sample_in_ours = json.loads("""{"id": 832047828167950336, "screen_name": "aarjavjain20039", "name": "Aarjav jain", "statuses_count": 22, "favourites_count": 11, "followers_count": 1, "friends_count": 45, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Feb 16 02:03:55 +0000 2017", "location": "saharanpur"}""")

  # sample_in_ours = json.loads("""{"id": 1020570515545747456, "screen_name": "WsifR", "name": "Wasif Risky", "statuses_count": 6, "favourites_count": 12, "followers_count": 21, "friends_count": 250, "listed_count": 0, "verified": false, "protected": false, "created_at": "Sat Jul 21 07:25:46 +0000 2018", "location": "Uttar Pradesh, India"}""")
  # print(sample_in_ours)
  
  # x = convert_ours_to_his(sample_in_ours)
  # print(x)
  # train_mask, valid_mask, test_mask = get_masks(581, 0.8, 0.1)
  # test_mask = [True if i in test_indices else False for i in range(mask_size)]
  # mask_te = np.array(test_mask)


  # a = np.array([0]).reshape(1,1)
  # y = np.array([0, 1]).reshape(1,2)

  # dataset = [Graph(x=x, a=a, y=y)]
  
  # x = np.array([22, 1, 45, 11, 0, 1, 1, 0, 1, 0]).reshape(1,10)
  # a = np.array([1]).reshape(1,1)
  # y = np.array([0, 1]).reshape(1,2)

  # dataset = [Graph(x=x, a=a, y=y)]
  # loader_te = SingleLoader(dataset, sample_weights=np.array([1]))
  # print(model.predict(loader_te.load(), steps=loader_te.steps_per_epoch))


  # print(sample_in_his)
  print("Prediction called")
  # x = np.array(sample_in_his)
  # x = x.reshape(1, 10)
  # print(x, x.shape)
  # print(model.predict(loader_te.load(), steps=loader_te.steps_per_epoch))
  
  a = ["""{"id": 725673298529509376, "screen_name": "afridi95531", "name": "Shahid", "statuses_count": 205, "favourites_count": 4487, "followers_count": 75, "friends_count": 635, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Apr 28 13:09:31 +0000 2016", "location": "Charlottetown, Prince Edward I"}""", """{"id": 832047828167950336, "screen_name": "aarjavjain20039", "name": "Aarjav jain", "statuses_count": 22, "favourites_count": 11, "followers_count": 1, "friends_count": 45, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Feb 16 02:03:55 +0000 2017", "location": "saharanpur"}""", """{"id": 947311093625573376, "screen_name": "AfghanSamoon", "name": "Afghan Samoon Society", "statuses_count": 1105, "favourites_count": 1245, "followers_count": 1745, "friends_count": 386, "listed_count": 3, "verified": false, "protected": false, "created_at": "Sun Dec 31 03:38:59 +0000 2017", "location": "Afghanistan"}""", """{"id": 1050771805802090497, "screen_name": "Anusiya12345", "name": "Anusiya", "statuses_count": 399, "favourites_count": 571, "followers_count": 673, "friends_count": 423, "listed_count": 0, "verified": false, "protected": false, "created_at": "Fri Oct 12 15:34:55 +0000 2018", "location": ""}""", """{"id": 847480576546189313, "screen_name": "AakashP28247364", "name": "Aakash Prajapat", "statuses_count": 29, "favourites_count": 1105, "followers_count": 27, "friends_count": 158, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Mar 30 16:08:09 +0000 2017", "location": ""}""", """{"id": 3909450376, "screen_name": "5472_nde", "name": "nde", "statuses_count": 49693, "favourites_count": 7373, "followers_count": 13591, "friends_count": 214, "listed_count": 206, "verified": false, "protected": false, "created_at": "Fri Oct 09 13:48:45 +0000 2015", "location": "United Kingdom"}""", """{"id": 1096707793069826048, "screen_name": "AnandKu96893531", "name": "Anand Kumar Dwivedi", "statuses_count": 52, "favourites_count": 374, "followers_count": 7, "friends_count": 146, "listed_count": 0, "verified": false, "protected": false, "created_at": "Sat Feb 16 09:48:08 +0000 2019", "location": "Uttar Pradesh, India"}""", """{"id": 1091007728519020544, "screen_name": "AmitmahajanAsr", "name": "Amit Mahajan Official 1.5k", "statuses_count": 3591, "favourites_count": 5012, "followers_count": 328, "friends_count": 1011, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Jan 31 16:18:07 +0000 2019", "location": "Amritsar, India"}""", """{"id": 941386009702490112, "screen_name": "AmitKum35359010", "name": "Amit Kumar", "statuses_count": 62, "favourites_count": 507, "followers_count": 12, "friends_count": 88, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Dec 14 19:14:49 +0000 2017", "location": "\u0909\u0928\u094d\u0928\u093e\u0935, \u092d\u093e\u0930\u0924"}""", """{"id": 107373151, "screen_name": "AmithGuptha", "name": "Amith Guptha", "statuses_count": 2984, "favourites_count": 2778, "followers_count": 223, "friends_count": 607, "listed_count": 1, "verified": false, "protected": false, "created_at": "Fri Jan 22 10:20:38 +0000 2010", "location": ""}""", """{"id": 1021812046567014400, "screen_name": "AkashMa30766119", "name": "chowkidar Akash Malviya", "statuses_count": 43, "favourites_count": 598, "followers_count": 10, "friends_count": 344, "listed_count": 0, "verified": false, "protected": false, "created_at": "Tue Jul 24 17:39:11 +0000 2018", "location": "mp"}""", """{"id": 847480576546189313, "screen_name": "AakashP28247364", "name": "Aakash Prajapat", "statuses_count": 29, "favourites_count": 1105, "followers_count": 27, "friends_count": 158, "listed_count": 0, "verified": false, "protected": false, "created_at": "Thu Mar 30 16:08:09 +0000 2017", "location": ""}"""]
  for e in a:
    print(convert_ours_to_his(json.loads(e)))

def convert_ours_to_his(d):
  m = [d["statuses_count"], d["followers_count"], d["friends_count"], d["favourites_count"], d["listed_count"]]

  # bg_image_false bg_image_true 
  # m.extend([0, 1])

  # verified_false verified_true 
  m.extend([int(d["verified"] == True), int(d["verified"] == False)])

  # protected_false protected_true
  m.extend([int(d["protected"] == True), int(d["protected"] == False)])

  return m

if __name__ == "__main__":
  main()