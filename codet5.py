from collections import namedtuple

from openprompt.plms import add_special_tokens
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

def load_plm():
    codet5_model = ModelClass(**{
        "config": T5Config, 
        "tokenizer": RobertaTokenizer, 
        "model": T5ForConditionalGeneration,
        "wrapper": T5TokenizerWrapper
    })

    model_class = codet5_model
    model_config = model_class.config.from_pretrained("Salesforce/codet5-base")
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    # if 'gpt' in model_name: # add pad token for gpt
    #     specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained("Salesforce/codet5-base", config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained("Salesforce/codet5-base")
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=None)

    # if 'opt' in model_name:
    #     tokenizer.add_bos_token=False
    
    return model, tokenizer, model_config, wrapper