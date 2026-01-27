from pymongo import MongoClient
from pathlib import Path

# -----------------------------
# MongoDB connection
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["snubhcvc"]
videos = db["videos"]

# -----------------------------
# Queries
# -----------------------------
query_stent_present = {
    "data.category.stent": 0
}

query_stent_absent = {
    "data.category.stent": -1
}

projection = {
    "_id": 0,
    "filename": 1
}

# -----------------------------
# Output paths
# -----------------------------
out_stent_present = Path("stent_present_filenames.txt")
out_stent_absent = Path("stent_absent_filenames.txt")

# -----------------------------
# Export function
# -----------------------------
def export_filenames(query, output_path):
    count = 0
    with open(output_path, "w") as f:
        cursor = videos.find(query, projection)
        for doc in cursor:
            filename = doc.get("filename")
            if filename:
                f.write(filename + "\n")
                count += 1
    return count

# -----------------------------
# Run export
# -----------------------------
count_present = export_filenames(query_stent_present, out_stent_present)
count_absent = export_filenames(query_stent_absent, out_stent_absent)

print("Export finished.")
print(f"Stent PRESENT (stent=0): {count_present} files -> {out_stent_present.resolve()}")
print(f"Stent ABSENT  (stent=-1): {count_absent} files -> {out_stent_absent.resolve()}")
