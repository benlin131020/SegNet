from os import path, chdir

chdir("./")
train_txt = open("dataset/train.txt", "w+")

for i in range(1, 200):
        train_path = "dataset/train/frame" + str(i) + ".jpg"
        trainannot_path = "dataset/trainannot/frame" + str(i) + ".png"
        if path.isfile(train_path) and path.isfile(trainannot_path):
                train_txt.write("Segnet/" + train_path + " " + "Segnet/" + trainannot_path + "\n")

train_txt.close()