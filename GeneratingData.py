from PIL import Image, ImageTk
import os
import shutil 
import json
import tkinter as tk
from zipfile import ZipFile

PATH_TO_IMAGEDATASET = r'imgDataset'
PATH_TO_14KFACES = r'C:\Users\harry\Desktop\Computing\real.zip'
PATH_TO_DATASET = r'annDataset'

# Create a smaller image dataset of unique faces from Microsoft DigiFace1M dataset
def createDataset(num, imagesperpax):
    numOfImages = 0

    with ZipFile(PATH_TO_14KFACES, 'a') as zipObj:

        # Get a list of filenames in zip object
        listOfFiles = zipObj.namelist()

        # Selecting unique faces 
        for (i, fileName) in enumerate(listOfFiles):
            if numOfImages == num:
                break
            
            if (i + 1) % imagesperpax == 0:
                zipObj.extract(fileName, path = PATH_TO_IMAGEDATASET)
                print(f"Extracted {fileName}")
                numOfImages += 1

    # Moving all files out of folders
    i = 0
    for dir in os.listdir(PATH_TO_IMAGEDATASET):
        for fileName in os.listdir(os.path.join(PATH_TO_IMAGEDATASET, dir)):
            src_path = os.path.join(PATH_TO_IMAGEDATASET, dir, fileName)
            dst_path = os.path.join(PATH_TO_IMAGEDATASET, str(i) + os.path.splitext(fileName)[1])
            shutil.move(src_path, dst_path)
            i += 1
        os.removedirs(os.path.join(PATH_TO_IMAGEDATASET, dir))

# Data Label Storage 
def writeTOJson(imageName, imagePath, gender, insults = [], compliments = []):
    gui_data = {imageName : {'path' : imagePath, 'gender': gender, 'insults' : insults, 'compliments' : compliments}}

    # Opening the JSON File
    try: 
        with open(os.path.join(PATH_TO_DATASET,'Annotations.json'),'r') as jsonFile:
            
            # Convert the JSON Data into Python object
            json_data = json.load(jsonFile)
            
            # Merge dict
            json_data.update(gui_data)
    except:
        json_data = gui_data

    with open(os.path.join(PATH_TO_DATASET,'Annotations.json'),'w') as jsonFile:

        json_object = json.dumps(json_data, indent = 4)
        jsonFile.write(json_object)


class GUI:
    def __init__(self):
        self.GUIData = []
        self.skip = False

    def closeWindow(self):
        # print("Close Window")
        
        # Getting Data
        gender = self.gender.get(1.0, tk.END).lower().replace("\n", "")
        compliments = self.compliments.get(1.0, tk.END).strip()
        insults = self.insults.get(1.0, tk.END).strip()
        # print(gender)
        # print(compliments)
        # print(insults)

        if gender:
            self.GUIData.append("Male" if gender[0] == "m" else "Female")
            if not compliments: self.GUIData.append([])
            else: self.GUIData.append(compliments.split("\n"))
            if not insults: self.GUIData.append([])
            else: self.GUIData.append(insults.split("\n"))
            # print(self.GUIData)
            self.window.destroy()
    
    def skip(self):
        self.skip = True
        self.window.destroy()
        print("Skipped")

    def summonGUI(self, src_file, imgno): 
        self.window = tk.Tk()
        self.window.geometry("600x380")
    
        # Image 
        image = Image.open(src_file)
        imageDisplayed = ImageTk.PhotoImage(image)

        imageLabel = tk.Label(image = imageDisplayed)

        # Image Number 
        imageNoLabel = tk.Label(text = "===== image " + imgno + " =====")

        # Gender Data
        genderLabel = tk.Label(
            text = "Gender: ",
        )
        self.gender = tk.Text(height = 1, width = 10)


        # Insults Data
        insultsLabel = tk.Label(
            text = "Insults: ",
        )
        self.insults = tk.Text(height = 5, width = 50)


        # Compliment Data
        complimentsLabel = tk.Label(
            text = "Compliments: "
        )
        self.compliments = tk.Text(height = 5, width = 50)


        # Complete Button
        completeBtn = tk.Button(
            self.window,
            text = "Confirm",
            width = 15,
            height = 2,
            bg = "yellow",
            command = self.closeWindow
        )

        # # Skip Button
        # skipBtn = tk.Button(
        #     self.window,
        #     text = "Skip",
        #     width = 15,
        #     height = 2,
        #     bg = "red",
        #     command = self.skip,
        # )

        # Packing everything
        imageLabel.pack(side = tk.RIGHT)
        imageNoLabel.pack()
        genderLabel.pack()
        self.gender.pack()
        insultsLabel.pack()
        self.insults.pack()
        complimentsLabel.pack()
        self.compliments.pack()
        completeBtn.pack()
        # skipBtn.pack()

        # Looping over the window 
        self.window.mainloop()
    
    def loopDataset(self):
        for imageName in os.listdir(PATH_TO_IMAGEDATASET):
            # GUI 
            newImgName = os.path.splitext(imageName)[0]
            src_path = os.path.join(PATH_TO_IMAGEDATASET, imageName)
            self.summonGUI(src_path, newImgName)

            if self.skip:
                self.skip = False
                continue

            # Retrieving data from GUI
            [gender, compliments, insults] = self.GUIData
            self.GUIData = []
            dst_path = os.path.join(PATH_TO_DATASET, newImgName + os.path.splitext(imageName)[1])
            shutil.move(src_path, dst_path)

            # Storing Data in json file 
            writeTOJson(newImgName, dst_path, gender, insults, compliments)

if __name__ == "__main__":
    # createDataset(1000, 5)

    guimain = GUI()
    guimain.loopDataset()


