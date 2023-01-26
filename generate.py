import torch
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer

class Generator(torch.nn.Module):
    def __init__(self,feature_size):
        super().__init__()
        self.tokens=AutoTokenizer.from_pretrained('facebook/opt-350m')
        self.config=AutoConfig.from_pretrained('facebook/opt-350m')
        self.text_model=OPTForCausalLM(self.config).from_pretrained('facebook/opt-350m')
        self.embeds=self.text_model.get_input_embeddings()
        hidden_size=self.embeds.embedding_dim
        self.features_to_embed=torch.nn.Sequential(torch.nn.Linear(feature_size+1,(feature_size+hidden_size+1)//2),torch.nn.GELU(),torch.nn.Linear((feature_size+hidden_size+1)//2,hidden_size))


    def forward(self,image_features,attitude,previous_text=None):
        features=torch.cat([image_features,attitude],dim=1)
        if previous_text==None:
            previous_text = ['' for i in range(len(attitude))]
        
        if not (len(image_features)==len(attitude)==len(previous_text)):
            raise ValueError('Batch size must be equal across all arguments')
        toks=self.tokens(previous_text,padding=True)
        previous_embed=self.embeds(torch.tensor(toks['input_ids']))
        starting_embed=self.features_to_embed(features).unsqueeze(1)
        start_seq=torch.cat([starting_embed,previous_embed],dim=1)
        return self.text_model(inputs_embeds=start_seq,attention_mask=torch.tensor([[1]+a for a in toks['attention_mask']]))
