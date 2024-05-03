import os
import re

river_ids = sorted(os.listdir("RiversideFrtInterior"))

# print(sorted(river_ids))

side_1_ids = []
side_2_ids = []

for id in river_ids:
    m = re.search('SD_(.+?)_', id)
    if m.group(1) == "01":
        side_1_ids.append(id)
    else:
        side_2_ids.append(id)

print(side_1_ids[:15])
print(side_2_ids[:5])
