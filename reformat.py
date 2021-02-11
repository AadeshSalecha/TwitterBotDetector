a = [1, 2, 7, 8]
global_dir = "C:/Users/Aadesh/Desktop/SNAM/TwitterBotDetector/"
for f in a:
  file_name = "timeline_folder_100_N" + str(f)
  with open(global_dir + file_name, "r") as inptr:
    with open(global_dir + "only_strings_" + file_name, "w") as outptr:
      inptr.readline()
      inptr.readline()
      raw_data = inptr.read()

      raw_data = raw_data.replace("timeline_data_", "")
      raw_data = raw_data.replace(".txt", "")
      raw_data = raw_data.replace("'", "")
      raw_data = raw_data.replace(", ", ",")

      outptr.write(raw_data[1:len(raw_data)-1])
