import torch
from peekingduck_process.preprocess_image import Preprocessor
from GAN.discern import Discerner
from GAN.generate import Generator
from dataloader import ImageOnlyDataLoader, ImageTextDataLoader

if torch.cuda.is_available():  
    device = "cuda:0"
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:  
    device = "cpu" 

print(device)
    

#Generator initialised with feature_size set to 512 as that is the size for jde, if we switch to a different peekingduck model, rmb to change
G=Generator(10).to(device)
D=Discerner(device=device).to(device)

#hyperparemeters, tune these
epochs=10000
max_sequence_length = 20
monte_carlo_iterations=10
sampling_temperature = 0.1
monte_carlo_sampling_temperature = 1.0
G_lr=0.01
D_lr=0.01

# model checkpointing parameters
epoch_checkpoint=100 # saves model every epoch_checkpoint epochs
data_content_folder = "" # leave blank for local system, use /content/NAISC/ for google colab
model_folder = r'/content/drive/MyDrive/NAISC_MODEL_CHECKPOINTS/' #PLEASE change accordingly to where you want to save your model


load_model_from_checkpoint = False


# training datasets
ImageOnlyDataset=ImageOnlyDataLoader(data_content_folder+"annDataset") 
ImageTextDataset=ImageTextDataLoader(data_content_folder+"annDataset/Annotations.json") 

#try different optimizers, should be a drag and drop replacement
G_optim=torch.optim.SGD(G.parameters(),lr=G_lr)
D_optim=torch.optim.SGD(D.parameters(),lr=D_lr)

initial_epoch = 0
if load_model_from_checkpoint:
    G_checkpoint = torch.load((model_folder + "generator.pt"))
    G.load_state_dict(G_checkpoint['model_state_dict'])
    G_optim.load_state_dict(G_checkpoint['optimizer_state_dict'])

    D_checkpoint = torch.load((model_folder + "discerner.pt"))
    D.load_state_dict(D_checkpoint['model_state_dict'])
    D_optim.load_state_dict(D_checkpoint['optimizer_state_dict'])
    
    initial_epoch = G_checkpoint['epoch']+1
    ImageOnlyDataset=ImageOnlyDataLoader(data_content_folder+"annDataset", data_queue=G_checkpoint['loader_queue'], random_generator=G_checkpoint['loader_rng'])
    ImageTextDataset=ImageTextDataLoader(data_content_folder+"annDataset/Annotations.json", data_queue=G_checkpoint['loader_queue'], random_generator=G_checkpoint['loader_rng']) 

torch.random.manual_seed(0)
#NOT batched because i dont care
for epoch in range(initial_epoch, epochs):
    print(f"EPOCH {epoch}")
    features=[]
    while features==[]:
        #reset to new proprocessor because jde has memory, luckily peeking duck does not need to redownload unlike SOME libraries
        preprocess=Preprocessor()
        image=next(ImageOnlyDataset)
        image, features=preprocess(image)
        if features:
            features=torch.tensor(features[0]).unsqueeze(0)
    attitudes=(2*torch.rand(1,1)-1)
    G_optim.zero_grad()
    toks, probs = G.forward(features, attitudes,max_length=max_sequence_length,temperature=sampling_temperature,return_probs=True)
    rewards=torch.tensor([])
    with torch.no_grad():
        for remaining_length in range(1,len(toks[0])):
            realness=torch.tensor([])
            for _ in range(monte_carlo_iterations):
                new_text=G.forward(features,attitudes,G.tokens.batch_decode([toks[0][:remaining_length]],skip_special_tokens=True),temperature=monte_carlo_sampling_temperature,return_probs=True,max_length=max_sequence_length-remaining_length-1,echo_input_text=True)[0]
                realness=torch.cat([realness,D.forward(image,G.tokens.batch_decode(new_text,skip_special_tokens=True),attitudes)[0]],dim=0)
            rewards=torch.cat([rewards,torch.mean(realness).unsqueeze(0)])
        final_text=G.tokens.batch_decode(toks,skip_special_tokens=True)
        rewards=torch.cat([rewards,D.forward(image,final_text,attitudes)[0]],dim=0)
    print(final_text[0])
    G_loss=-torch.sum(rewards*torch.log(probs[0]))
    print("Generator loss:", G_loss)
    G_loss.backward()
    G_optim.step()
    D_optim.zero_grad()
    d_image,d_text,d_attitude=next(ImageTextDataset)
    D_loss=-(D.forward([d_image],[d_text],[d_attitude])[0]-D.forward(image,final_text,attitudes)[0])
    print("Discerner loss:", D_loss)
    D_loss.backward()
    D_optim.step()

    if (epoch+1) % epoch_checkpoint == 0 or epoch == epochs:
        torch.save({
            'epoch': epoch,
            'model_state_dict': G.state_dict(),
            'optimizer_state_dict': G_optim.state_dict(),
            'loss': G_loss,
            'loader_queue': ImageOnlyDataset.data_queue,
            'loader_rng': ImageOnlyDataset.rng
        }, (model_folder + "generator.pt"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': D.state_dict(),
            'optimizer_state_dict': D_optim.state_dict(),
            'loss': D_loss,
            'loader_queue': ImageTextDataset.data_queue,
            'loader_rng': ImageTextDataset.rng
        }, (model_folder + "discerner.pt"))
