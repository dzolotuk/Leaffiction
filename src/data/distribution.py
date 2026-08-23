import sys
import os
import matplotlib.pyplot as plt

directory = sys.argv[1]

content = os.listdir(directory)
result = {}

for element in content:
    path = os.path.join(directory, element)
    if os.path.isdir(path):
        files = os.listdir(path)
        nb_images = 0
        for file in files:
            name, extension = os.path.splitext(file)
            if extension in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
                nb_images += 1
        result[element] = nb_images

label = list(result.keys())
val = list(result.values())
colors = plt.cm.Set2.colors
plant_name = os.path.basename(os.path.normpath(directory))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(plant_name + " class distribution", fontsize=14)

ax1.pie(val, labels=label, autopct="%1.1f%%")
ax1.set_title("Distribution per %")

ax2.bar(label, val, color=colors[:len(label)])
ax2.set_title("Number of images")

plt.xticks(rotation=30)
plt.show()
