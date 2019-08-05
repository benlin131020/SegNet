from os import path, chdir

chdir("./")
val_txt = open("dataset/val.txt", "w+")

for i in range(1, 200):
        val_path = "dataset/val/frame" + str(i) + ".jpg"
        valannot_path = "dataset/valannot/frame" + str(i) + ".png"
        if path.isfile(val_path) and path.isfile(valannot_path):
                val_txt.write("Segnet/" + val_path + " " + "Segnet/" + valannot_path + "\n")

val_txt.close()