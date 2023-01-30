from PIL import Image
import random
import json

class ImageOnlyDataLoader:
    # used for generator
    def __init__(self, JSONfilepath, replaceDirSepChar=False):
        # set replaceDirSepChar to True on non-Windows systems to replace \\ to /
        self.idx = 0
        self.imgdata = []
        with open(JSONfilepath, "r") as file:
            data = json.load(file)
            if replaceDirSepChar:
                self.imgdata = [Image.open(data[key]['path'].replace("\\", "/")) for key in data.keys()]
            else:
                self.imgdata = [Image.open(data[key]['path']) for key in data.keys()]
        random.shuffle(self.imgdata)
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx == len(self.imgdata):
            self.idx = 0
            random.shuffle(self.imgdata)
        self.idx += 1
        return self.imgdata[self.idx-1]
    def __len__(self):
        return len(self.imgdata)
    next = __next__

class ImageTextDataLoader:
    # used for generator
    def __init__(self, JSONfilepath, replaceDirSepChar=False, skipUnratedStatements=False):
        # set replaceDirSepChar to True on non-Windows systems to replace \\ to /
        self.idx = 0
        self.imgtxtdata = []
        with open(JSONfilepath, "r") as file:
            data = json.load(file)
            for key in data.keys():
                data_dict = data[key]
                imgpath = data_dict['path'].replace("\\", "/") if replaceDirSepChar else data_dict['path']
                for insult in data_dict['insults']:
                    if "~" not in insult and not skipUnratedStatements:
                        raise ValueError("Unrated statment occured, use ~ to rate statements")
                    elif "~" not in insult:
                        continue
                    statement, attitude = insult.split("~")
                    attitude = -float(attitude)
                    self.imgtxtdata.append((Image.open(imgpath), statement, attitude))
                for compliment in data_dict['compliments']:
                    if "~" not in compliment and not skipUnratedStatements:
                        raise ValueError("Unrated statment occured, use ~ to rate statements")
                    elif "~" not in compliment:
                        continue
                    statement, attitude = compliment.split("~")
                    attitude = float(attitude)
                    self.imgtxtdata.append((Image.open(imgpath), statement, attitude))
        random.shuffle(self.imgtxtdata)
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx == len(self.imgtxtdata):
            self.idx = 0
            random.shuffle(self.imgtxtdata)
        self.idx += 1
        return self.imgtxtdata[self.idx-1]
    def __len__(self):
        return len(self.imgtxtdata)
    next = __next__

if __name__ == "__main__":
    ImageOnlyDataset = ImageOnlyDataLoader("annDataset/annotations.json", replaceDirSepChar=True)
    print(len(ImageOnlyDataset))
    ImageTextDataset = ImageTextDataLoader("annDataset/annotations.json", replaceDirSepChar=True, skipUnratedStatements=True)
    print(len(ImageTextDataset))
