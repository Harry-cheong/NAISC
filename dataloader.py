from PIL import Image
import random
import json

class ImageOnlyDataLoader:
    # used for generator
    def __init__(self, JSONfilepath, total_epochs, curr_idx=0, replaceDirSepChar=False):
        # set replaceDirSepChar to True on non-Windows systems to replace \\ to /
        self.idx = curr_idx
        dataset = []
        with open(JSONfilepath, "r") as file:
            data = json.load(file)
            if replaceDirSepChar:
                dataset = [Image.open(data[key]['path'].replace("\\", "/")) for key in data.keys()]
            else:
                dataset = [Image.open(data[key]['path']) for key in data.keys()]
        self.imgdata = []
        for _ in range(len(dataset), total_epochs, len(dataset)):
            random.shuffle(dataset)
            self.imgdata += dataset
        random.shuffle(dataset)
        cut_dataset = dataset[:(total_epochs % len(dataset))]
        self.imgdata += cut_dataset
    def __iter__(self):
        return self
    def __next__(self):
        self.idx += 1
        try:
            return self.imgdata[self.idx-1]
        except IndexError:
            raise StopIteration
    def __len__(self):
        return len(self.imgdata)
    next = __next__

class ImageTextDataLoader:
    # used for discerner
    def __init__(self, JSONfilepath, total_epochs, curr_idx=0, replaceDirSepChar=False, skipUnratedStatements=False):
        # set replaceDirSepChar to True on non-Windows systems to replace \\ to /
        self.idx = curr_idx
        dataset = []
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
                    dataset.append((Image.open(imgpath), statement, attitude))
                for compliment in data_dict['compliments']:
                    if "~" not in compliment and not skipUnratedStatements:
                        raise ValueError("Unrated statment occured, use ~ to rate statements")
                    elif "~" not in compliment:
                        continue
                    statement, attitude = compliment.split("~")
                    attitude = float(attitude)
                    dataset.append((Image.open(imgpath), statement, attitude))
        self.imgtxtdata = []
        for _ in range(len(dataset), total_epochs, len(dataset)):
            random.shuffle(dataset)
            self.imgtxtdata += dataset
        random.shuffle(dataset)
        self.imgtxtdata += dataset[:(total_epochs % len(dataset))]
    def __iter__(self):
        return self
    def __next__(self):
        self.idx += 1
        try:
            return self.imgtxtdata[self.idx-1]
        except IndexError:
            raise StopIteration
    def __len__(self):
        return len(self.imgtxtdata)
    next = __next__

if __name__ == "__main__":
    ImageOnlyDataset = ImageOnlyDataLoader("annDataset/annotations.json", 1000, replaceDirSepChar=True)
    print(len(ImageOnlyDataset))
    ImageTextDataset = ImageTextDataLoader("annDataset/annotations.json", 1000, replaceDirSepChar=True, skipUnratedStatements=True)
    print(len(ImageTextDataset))
