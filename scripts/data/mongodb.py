from pymongo import MongoClient


client = MongoClient("mongodb://localhost:27017/")
db = client["snubhcvc"]
videos = db["videos"]


# Query: "data" exists, is an array, and has at least one item
query = {
    "data": {
        "$exists": True,
        # "$type": "array",
        "$ne": []  # Not equal to empty list
    }
}

results = list(videos.find(query))
print(len(results))  # 265250



# stent: 1132 coro_wire: 859 ic_device: 244 ied: 367 stern_wire: 199 collat: 289 
# example of category keys:
#   ['balloon_catheter' 'collat' 'coro_wire' 'coronary_wire' 'ic_device' 'id'
#    'ied' 'is_valid' 'left_right' 'pacemaker' 'stent' 'stern_wire']

multi_label_data = []
for res in results:
    category = res["data"]["category"]
    # if "stent" in category:
    if len(category.keys()) > 2:
        multi_label_data.append(res)

print(len(multi_label_data))  # 2160






num_stent_data = 0  # 1270
num_coro_wire_data = 0  # 884
num_ic_device_data = 0  # 294
num_ied_data = 0  # 378
num_stern_wire_data = 0  # 214
num_collat_data = 0  # 333

for data in multi_label_data:
    if "stent" in data["data"]["category"]:
        num_stent_data += 1
        # print(data)
    if "coro_wire" in data["data"]["category"]:
        num_coro_wire_data += 1
    if "ic_device" in data["data"]["category"]:
        num_ic_device_data += 1
    if "ied" in data["data"]["category"]:
        num_ied_data += 1
    if "stern_wire" in data["data"]["category"]:
        num_stern_wire_data += 1
    if "collat" in data["data"]["category"]:
        num_collat_data += 1

print(num_stent_data, num_coro_wire_data, num_ic_device_data, num_ied_data, num_stern_wire_data, num_collat_data)