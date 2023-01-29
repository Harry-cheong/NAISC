import torch
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
import torch.nn.functional as F

class Generator(torch.nn.Module):
    def __init__(self,feature_size):
        super().__init__()
        self.tokens=AutoTokenizer.from_pretrained('facebook/opt-350m', padding_side='left')
        self.config=AutoConfig.from_pretrained('facebook/opt-350m')
        self.text_model=OPTForCausalLM(self.config).from_pretrained('facebook/opt-350m')
        self.embeds=self.text_model.get_input_embeddings()
        hidden_size=self.embeds.embedding_dim
        self.features_to_embed=torch.nn.Sequential(torch.nn.Linear(feature_size+1,(feature_size+hidden_size+1)//2),torch.nn.GELU(),torch.nn.Linear((feature_size+hidden_size+1)//2,hidden_size))
    
    def forward(self, image_features, attitude, starting_text = None, max_length = 10, temperature = 1.0, return_probs = False):

        if starting_text==None:
            starting_text = ['']*len(attitude)

        if not (len(image_features)==len(attitude)==len(starting_text)):
            raise ValueError('Batch size must be equal across all arguments')

        out_text=starting_text
        all_new_toks=torch.tensor([[]]*len(attitude),dtype=torch.int)
        out_probs=[]
        for _ in range(max_length):
            in_embeds, mask=self._generate_embeddings_and_masks(out_text,image_features,attitude)
            new_tok_logits=self.text_model(inputs_embeds=in_embeds,attention_mask=mask).logits[:,-1,:]
            tok_probs=F.gumbel_softmax(new_tok_logits,tau=temperature,dim=-1)
            new_toks=torch.distributions.categorical.Categorical(probs=tok_probs).sample().unsqueeze(1)
            if return_probs:
                out_probs.append(torch.gather(tok_probs, 1, new_toks))
            all_new_toks=torch.cat([all_new_toks,new_toks],dim=1)
            new_text=self.tokens.batch_decode(new_toks)
            out_text=[old+new for old, new in zip(out_text,new_text)]
            if all((self.tokens.eos_token_id in toks) for toks in all_new_toks):
                break
        out_text=[text.split(self.tokens.eos_token)[0] for text in out_text]
        if not return_probs:
            return out_text
        out_probs=[prob[:len(toks)] for toks,prob in zip(all_new_toks,torch.cat(out_probs,dim=1))]
        return all_new_toks, out_probs
        
            
    def _generate_embeddings_and_masks(self, previous_text, image_features, attitude):
        features=torch.cat([image_features,attitude],dim=1)
        toks=self.tokens(previous_text,padding=True,return_tensors='pt')
        previous_embed=self.embeds(toks['input_ids'])
        starting_embed=self.features_to_embed(features).unsqueeze(1)
        #return previous_embed, toks['attention_mask']                      this is for if you dont want the image or attitude to affect text generation
        return torch.cat([starting_embed,previous_embed],dim=1), torch.cat([toks['attention_mask'],torch.ones(len(attitude),1)],dim=1)
        


if __name__=='__main__':
    x=Generator(2)
    print(x.forward(torch.tensor([[1,2],[3,4]],dtype=torch.float),torch.tensor([[1],[1]]),['Hello, pleased to meet you','I hope you die painfully'],return_probs=True))
