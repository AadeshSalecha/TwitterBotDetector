# this script already exists in project/srivasta
# run it with python3 user_checks.py

import os

users_to_check = ["ghayalhabib87", "HRMinistryPak", "rachellFlt", "jaipurjyotish01", "Shruti_quotes", "annechuaa"]

for user in users_to_check:
  if not os.path.exists("./timeline_folder_100_N2/" + "timeline_data_" + user + ".txt"):
    print(user)
  else:
    with open("./timeline_folder_100_N2/" + "timeline_data_" + user + ".txt") as inptr:
      print(user, " exists")
      print(inptr.read()[:100])
