import torch
from discern import Discerner
from generate import Generator
from preprocess_image import Preprocessor
from PIL import Image
import itertools

ImageOnlyDataset=itertools.repeat(Image.open('people.webp')) #Random images contating one person, should output PIL.Image object
ImageTextDataset=iter([]) #A random image with one person inside and one random compliment/insult, should output a tuple with (image, statement, attitude_score)
#DataSet treated as an iterator, with each call to its __next__ method yielding one data point, refer to Python documentation as to how __next__ should be implemented

#Generator initialised with feature_size set to 512 as that is the size for jde, if we switch to a different peekingduck model, rmb to change
G=Generator(512)
D=Discerner()

#hyperparemeters, tune these
epochs=10000
max_sequence_length = 20
monte_carlo_iterations=10
sampling_temperature = 0.1
monte_carlo_sampling_temperature = 1.0
G_lr=0.01
D_lr=0.01

#try different optimizers, should be a drag and drop replacement
G_optim=torch.optim.SGD(G.parameters(),lr=G_lr)
D_optim=torch.optim.SGD(D.parameters(),lr=D_lr)
torch.random.manual_seed(0)
#NOT batched because i dont care
for epoch in range(epochs):
    features=[]
    while features==[]:
        #reset to new proprocessor because jde has memory, luckily peeking duck does not need to redownload unlike SOME libraries
        preprocess=Preprocessor()
        image=next(ImageOnlyDataset)
        #need to run twice cuz jde is optimzed for video and can't capture people until fed two frames
        for i in range(2):
            features=preprocess(image)[1]
            if features:
                features=torch.tensor(features[0]).unsqueeze(0)
    attitudes=2*torch.rand(1,1)-1
    G_optim.zero_grad()
    toks, probs = G.forward(features, attitudes,max_length=max_sequence_length,temperature=sampling_temperature,return_probs=True)
    rewards=torch.tensor([])
    with torch.no_grad():
        for remaining_length in range(len(toks[0])-1):
            realness=torch.tensor([])
            for _ in range(monte_carlo_iterations):
                new_text=G.forward(features,attitudes,G.tokens.batch_decode(toks[:,:remaining_length],skip_special_tokens=True),temperature=monte_carlo_sampling_temperature,return_probs=True,max_length=max_sequence_length-remaining_length-1,echo_input_text=True)[0]
                realness=torch.cat([realness,D.forward([image],G.tokens.batch_decode(new_text,skip_special_tokens=True),attitudes)[0]],dim=0)
            torch.cat([rewards,torch.mean(realness)])
        rewards=torch.cat([rewards,D.forward([image],G.tokens.batch_decode(toks,skip_special_tokens=True),attitudes)[0]],dim=0)
    loss=-torch.sum(rewards*torch.log(probs[0]))
    loss.backward()
    G_optim.step()
    D_optim.zero_grad()
    image,text,attitude=next(ImageTextDataset)
    loss=-(D.forward([image],[text],[attitude])[0]-D.forward([image],G.tokens.batch_decode(toks,skip_special_tokens=True),attitudes)[0])
    loss.backward()
    D_optim.step()
    




    
    