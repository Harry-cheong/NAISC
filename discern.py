import torch
import clip
from PIL import Image
from sgnlp.models.sentic_gcn import(
    SenticGCNBertConfig,
    SenticGCNBertModel,
    SenticGCNBertEmbeddingConfig,
    SenticGCNBertEmbeddingModel,
    SenticGCNBertTokenizer,
    SenticGCNBertPreprocessor
)
from string import punctuation
import re
import torch
import numpy as np

class Discerner(torch.nn.Module):
    def __init__(self):
        
        super().__init__()
        #initialise all the models
        tokenizer = SenticGCNBertTokenizer.from_pretrained("bert-base-uncased")
        config = SenticGCNBertConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/config.json")
        self.sentiment_model = SenticGCNBertModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticgcn_bert/pytorch_model.bin", config=config)
        embed_config = SenticGCNBertEmbeddingConfig.from_pretrained("bert-base-uncased")
        embed_model = SenticGCNBertEmbeddingModel.from_pretrained("bert-base-uncased", config=embed_config)
        self.sentimentpreprocessor = SenticGCNBertPreprocessor(tokenizer=tokenizer, embedding_model=embed_model, senticnet="https://storage.googleapis.com/sgnlp/models/sentic_gcn/senticnet.pickle", device="cpu")
        self.clip_model, self.imagepreprocessor = clip.load('ViT-B/32', jit=False)
        self.imagetextfeatures = torch.nn.Sequential(torch.nn.Linear(512*2,768), torch.nn.GELU(), torch.nn.Linear(768,512), torch.nn.GELU())
        self.sentiment_gru = torch.nn.GRU(input_size=3, hidden_size=3,batch_first=True)
        self.sentiment_attitude_corr=torch.nn.Sequential(torch.nn.Linear(4,258),torch.nn.GELU(),torch.nn.Linear(258,512),torch.nn.GELU())
        self.discern=torch.nn.Sequential(torch.nn.Linear(1024,512),torch.nn.GELU(),torch.nn.Linear(512,1))

    def forward(self, image, statement, attitude):
        attitude=torch.tensor([[a] for a in attitude])
        if not (len(image)==len(statement)==len(attitude)):
            raise ValueError('Batch size for all arguments must be the same')
        
        image_features = self.clip_model.encode_image(torch.cat([self.imagepreprocessor(im).unsqueeze(0) for im in image]))
        text_features = self.clip_model.encode_text(clip.tokenize(statement))
        clip_features = self.imagetextfeatures(torch.cat((image_features,text_features),dim=1))

        inputs=[]
        inputs_aspect_count=[]
        for stat in statement:
            stat = stat.lower()
            for p in punctuation:
                stat = stat.replace(p, f' {p} ')
            stat=re.sub(r"\s+", " ", stat)
            
            # use each word as 1 aspect (except punctuation)
            stat_split = stat.split(' ')
            aspects = [word for word in stat_split if word not in punctuation]

            # add to inputs
            inputs.append( {'sentence': stat, 'aspects': list(dict.fromkeys(aspects))} )
            inputs_aspect_count.append(len(aspects))
         
        processed_indices = self.sentimentpreprocessor(inputs)[1]
        probabilities = self.sentiment_model(processed_indices).logits
        sentiment_packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(torch.nn.utils.rnn.pad_sequence(torch.split(probabilities, inputs_aspect_count, dim=0),batch_first=True),batch_first=True,enforce_sorted=False,lengths=inputs_aspect_count)
        processed_sentiments=[]
        for tensor, index in zip(*torch.nn.utils.rnn.pad_packed_sequence(self.sentiment_gru(sentiment_packed_outputs)[0],batch_first=True)):
            processed_sentiments.append(tensor[index-1].unsqueeze(0))
        processed_sentiments=torch.cat([torch.cat(processed_sentiments),attitude],dim=1)
        all_features=torch.cat((clip_features,self.sentiment_attitude_corr(processed_sentiments)),dim=1)
        return self.discern(all_features)
        
image=Image.open('Snakes_Diversity.jpg')
print(Discerner().forward([image,image],['snake is good','snake is very bad'],[1,-1]))