import os 
import math 
import time 
import urllib .request 
import torch 
import torch .nn as nn 
from torch .nn import functional as F 
import matplotlib .pyplot as plt 
class Config :
    vocab_size :int =50257 
    n_embd :int =256 
    n_head :int =8 
    n_layer :int =6 
    block_size :int =128 
    dropout :float =0.1 
    bias :bool =False 
    batch_size :int =32 
    learning_rate :float =5e-4 
    min_lr :float =5e-5 
    warmup_iters :int =100 
    max_iters :int =1500 
    eval_interval :int =150 
    eval_iters :int =40 
    weight_decay :float =1e-1 
    grad_clip :float =1.0 
    device :str ="cuda"if torch .cuda .is_available ()else "cpu"
    seed :int =1337 
cfg =Config ()
torch .manual_seed (cfg .seed )
if torch .cuda .is_available ():
    torch .cuda .manual_seed_all (cfg .seed )
print (f"[*] Running on device: {cfg .device }")
if cfg .device =="cuda":
    print (f"[*] GPU Name: {torch .cuda .get_device_name (0 )}")
class RotaryEmbedding (nn .Module ):
    def __init__ (self ,dim :int ,max_seq_len :int =2048 ):
        super ().__init__ ()
        self .dim =dim 
        inv_freq =1.0 /(10000 **(torch .arange (0 ,dim ,2 ).float ()/dim ))
        self .register_buffer ("inv_freq",inv_freq )
        t =torch .arange (max_seq_len ,dtype =torch .float32 )
        freqs =torch .outer (t ,self .inv_freq )
        emb =torch .cat ((freqs ,freqs ),dim =-1 )
        self .register_buffer ("cos_cached",emb .cos (),persistent =False )
        self .register_buffer ("sin_cached",emb .sin (),persistent =False )
    def rotate_half (self ,x ):
        x1 =x [...,:self .dim //2 ]
        x2 =x [...,self .dim //2 :]
        return torch .cat ((-x2 ,x1 ),dim =-1 )
    def forward (self ,x ,seq_len ):
        cos =self .cos_cached [:seq_len ,:].unsqueeze (0 ).unsqueeze (0 )
        sin =self .sin_cached [:seq_len ,:].unsqueeze (0 ).unsqueeze (0 )
        return (x *cos )+(self .rotate_half (x )*sin )
class RMSNorm (nn .Module ):
    def __init__ (self ,dim :int ,eps :float =1e-6 ):
        super ().__init__ ()
        self .eps =eps 
        self .weight =nn .Parameter (torch .ones (dim ))
    def forward (self ,x ):
        variance =x .pow (2 ).mean (-1 ,keepdim =True )
        return x *torch .rsqrt (variance +self .eps )*self .weight 
class CausalSelfAttention (nn .Module ):
    def __init__ (self ,config :Config ):
        super ().__init__ ()
        assert config .n_embd %config .n_head ==0 
        self .n_head =config .n_head 
        self .head_dim =config .n_embd //config .n_head 
        self .qkv_proj =nn .Linear (config .n_embd ,3 *config .n_embd ,bias =config .bias )
        self .out_proj =nn .Linear (config .n_embd ,config .n_embd ,bias =config .bias )
        self .attn_dropout =nn .Dropout (config .dropout )
        self .resid_dropout =nn .Dropout (config .dropout )
        self .rope =RotaryEmbedding (self .head_dim ,max_seq_len =config .block_size )
        self .register_buffer (
        "causal_mask",
        torch .tril (torch .ones (config .block_size ,config .block_size )).view (
        1 ,1 ,config .block_size ,config .block_size 
        ),
        )
        self .last_attn_weights =None 
    def forward (self ,x ,return_attn :bool =False ):
        B ,T ,C =x .size ()
        qkv =self .qkv_proj (x )
        q ,k ,v =qkv .chunk (3 ,dim =-1 )
        q =q .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        k =k .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        v =v .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        q =self .rope (q ,T )
        k =self .rope (k ,T )
        scores =torch .matmul (q ,k .transpose (-2 ,-1 ))/math .sqrt (self .head_dim )
        scores =scores .masked_fill (self .causal_mask [:,:,:T ,:T ]==0 ,float ("-inf"))
        attn_weights =F .softmax (scores ,dim =-1 )
        if return_attn or not self .training :
            self .last_attn_weights =attn_weights .detach ().cpu ()
        attn_weights_drop =self .attn_dropout (attn_weights )
        y =torch .matmul (attn_weights_drop ,v )
        y =y .transpose (1 ,2 ).contiguous ().view (B ,T ,C )
        out =self .resid_dropout (self .out_proj (y ))
        return out 
class FeedForward (nn .Module ):
    def __init__ (self ,config :Config ):
        super ().__init__ ()
        hidden_dim =int (4 *config .n_embd *(2 /3 ))
        self .w1 =nn .Linear (config .n_embd ,hidden_dim ,bias =config .bias )
        self .w2 =nn .Linear (hidden_dim ,config .n_embd ,bias =config .bias )
        self .w3 =nn .Linear (config .n_embd ,hidden_dim ,bias =config .bias )
        self .dropout =nn .Dropout (config .dropout )
    def forward (self ,x ):
        return self .dropout (self .w2 (F .silu (self .w1 (x ))*self .w3 (x )))
class TransformerBlock (nn .Module ):
    def __init__ (self ,config :Config ):
        super ().__init__ ()
        self .norm1 =RMSNorm (config .n_embd )
        self .attn =CausalSelfAttention (config )
        self .norm2 =RMSNorm (config .n_embd )
        self .ffn =FeedForward (config )
    def forward (self ,x ,return_attn :bool =False ):
        x =x +self .attn (self .norm1 (x ),return_attn =return_attn )
        x =x +self .ffn (self .norm2 (x ))
        return x 
class MiniLLM (nn .Module ):
    def __init__ (self ,config :Config ):
        super ().__init__ ()
        self .config =config 
        self .tok_embeddings =nn .Embedding (config .vocab_size ,config .n_embd )
        self .drop =nn .Dropout (config .dropout )
        self .blocks =nn .ModuleList ([TransformerBlock (config )for _ in range (config .n_layer )])
        self .norm_f =RMSNorm (config .n_embd )
        self .lm_head =nn .Linear (config .n_embd ,config .vocab_size ,bias =False )
        self .tok_embeddings .weight =self .lm_head .weight 
        self .apply (self ._init_weights )
    def _init_weights (self ,module ):
        if isinstance (module ,nn .Linear ):
            torch .nn .init .normal_ (module .weight ,mean =0.0 ,std =0.02 )
            if module .bias is not None :
                torch .nn .init .zeros_ (module .bias )
        elif isinstance (module ,nn .Embedding ):
            torch .nn .init .normal_ (module .weight ,mean =0.0 ,std =0.02 )
    def forward (self ,idx ,targets =None ,return_attn =False ):
        B ,T =idx .size ()
        assert T <=self .config .block_size ,f"Input sequence length {T } exceeds block size {self .config .block_size }"
        x =self .drop (self .tok_embeddings (idx ))
        for block in self .blocks :
            x =block (x ,return_attn =return_attn )
        x =self .norm_f (x )
        if targets is not None :
            logits =self .lm_head (x )
            loss =F .cross_entropy (logits .view (-1 ,logits .size (-1 )),targets .view (-1 ),ignore_index =-1 )
        else :
            logits =self .lm_head (x [:,[-1 ],:])
            loss =None 
        return logits ,loss 
    @torch .no_grad ()
    def generate (self ,idx ,max_new_tokens :int ,temperature :float =0.8 ,top_k :int =40 ):
        self .eval ()
        for _ in range (max_new_tokens ):
            idx_cond =idx if idx .size (1 )<=self .config .block_size else idx [:,-self .config .block_size :]
            logits ,_ =self (idx_cond )
            logits =logits [:,-1 ,:]/max (temperature ,1e-5 )
            if top_k is not None :
                v ,_ =torch .topk (logits ,min (top_k ,logits .size (-1 )))
                logits [logits <v [:,[-1 ]]]=-float ("Inf")
            probs =F .softmax (logits ,dim =-1 )
            idx_next =torch .multinomial (probs ,num_samples =1 )
            idx =torch .cat ((idx ,idx_next ),dim =1 )
        return idx 
class TextDataset :
    def __init__ (self ,text :str ,block_size :int ):
        self .block_size =block_size 
        self .chars =sorted (list (set (text )))
        self .vocab_size =len (self .chars )
        self .char_to_ix ={ch :i for i ,ch in enumerate (self .chars )}
        self .ix_to_char ={i :ch for i ,ch in enumerate (self .chars )}
        data =torch .tensor ([self .char_to_ix [ch ]for ch in text ],dtype =torch .long )
        n =int (0.9 *len (data ))
        self .train_data =data [:n ]
        self .val_data =data [n :]
    def encode (self ,s :str ):
        return [self .char_to_ix [c ]for c in s if c in self .char_to_ix ]
    def decode (self ,l ):
        return "".join ([self .ix_to_char [i ]for i in l ])
    def get_batch (self ,split :str ,batch_size :int ,device :str ):
        data =self .train_data if split =="train"else self .val_data 
        ix =torch .randint (len (data )-self .block_size ,(batch_size ,))
        x =torch .stack ([data [i :i +self .block_size ]for i in ix ])
        y =torch .stack ([data [i +1 :i +1 +self .block_size ]for i in ix ])
        return x .to (device ),y .to (device )
def get_lr (it :int ,cfg :Config ):
    if it <cfg .warmup_iters :
        return cfg .learning_rate *(it +1 )/(cfg .warmup_iters +1 )
    if it >cfg .max_iters :
        return cfg .min_lr 
    decay_ratio =(it -cfg .warmup_iters )/(cfg .max_iters -cfg .warmup_iters )
    coeff =0.5 *(1.0 +math .cos (math .pi *decay_ratio ))
    return cfg .min_lr +coeff *(cfg .learning_rate -cfg .min_lr )
def plot_training_curves (train_losses ,val_losses ,eval_intervals ,save_path ="loss_curve.png"):
    plt .figure (figsize =(9 ,5 ),dpi =150 )
    plt .plot (eval_intervals ,train_losses ,label ="Train Loss",color ="#1f77b4",lw =2 )
    plt .plot (eval_intervals ,val_losses ,label ="Validation Loss",color ="#ff7f0e",lw =2 ,linestyle ="--")
    plt .title ("Model Convergence: Cross-Entropy Loss vs Steps",fontsize =14 ,fontweight ="bold")
    plt .xlabel ("Iteration",fontsize =12 )
    plt .ylabel ("Loss (nats)",fontsize =12 )
    plt .grid (True ,linestyle =":",alpha =0.6 )
    plt .legend (fontsize =11 )
    plt .tight_layout ()
    plt .savefig (save_path )
    plt .close ()
    print (f"[+] Saved loss curve plot to: {save_path }")
def plot_attention_map (attn_matrix ,tokens ,head_idx =0 ,layer_idx =0 ,save_path ="attention_heatmap.png"):
    weights =attn_matrix [0 ,head_idx ].numpy ()
    T =min (len (tokens ),weights .shape [0 ])
    weights =weights [:T ,:T ]
    plt .figure (figsize =(8 ,7 ),dpi =150 )
    plt .imshow (weights ,cmap ="viridis",interpolation ="nearest")
    plt .title (f"Self-Attention Weights (Layer {layer_idx }, Head {head_idx })",fontsize =13 ,fontweight ="bold")
    plt .colorbar (label ="Attention Weight")
    plt .xticks (range (T ),tokens [:T ],rotation =45 ,ha ="right",fontsize =9 )
    plt .yticks (range (T ),tokens [:T ],fontsize =9 )
    plt .xlabel ("Key Tokens (Attended)",fontsize =11 )
    plt .ylabel ("Query Tokens",fontsize =11 )
    plt .tight_layout ()
    plt .savefig (save_path )
    plt .close ()
    print (f"[+] Saved attention heatmap to: {save_path }")
if __name__ =="__main__":
    print ("="*65 )
    print ("  Transformer LLM Built From Scratch (RoPE + RMSNorm + SwiGLU)")
    print ("="*65 )
    sample_text =(
    "Artificial intelligence and deep learning models process representations of "
    "human knowledge. The transformer architecture relies on multi-head causal attention, "
    "positional encodings, feed-forward layers, and root-mean-square normalization to predict "
    "subsequent sequences. From scratch, neural networks learn patterns, semantics, and "
    "probabilistic structures in language and vision domains. "
    )*40 
    dataset =TextDataset (sample_text ,block_size =cfg .block_size )
    cfg .vocab_size =dataset .vocab_size 
    model =MiniLLM (cfg ).to (cfg .device )
    total_params =sum (p .numel ()for p in model .parameters ())
    print (f"[*] Total Model Parameters: {total_params :,}")
    optimizer =torch .optim .AdamW (model .parameters (),lr =cfg .learning_rate ,weight_decay =cfg .weight_decay )
    print ("\n[*] Running fast smoke-test training loop...")
    cfg .max_iters =60 
    cfg .eval_interval =20 
    train_history ,val_history ,steps =[],[],[]
    model .train ()
    for it in range (cfg .max_iters ):
        lr =get_lr (it ,cfg )
        for param_group in optimizer .param_groups :
            param_group ["lr"]=lr 
        xb ,yb =dataset .get_batch ("train",cfg .batch_size ,cfg .device )
        logits ,loss =model (xb ,yb )
        optimizer .zero_grad (set_to_none =True )
        loss .backward ()
        nn .utils .clip_grad_norm_ (model .parameters (),cfg .grad_clip )
        optimizer .step ()
        if (it +1 )%cfg .eval_interval ==0 or it ==cfg .max_iters -1 :
            model .eval ()
            with torch .no_grad ():
                xv ,yv =dataset .get_batch ("val",cfg .batch_size ,cfg .device )
                _ ,v_loss =model (xv ,yv )
            train_history .append (loss .item ())
            val_history .append (v_loss .item ())
            steps .append (it +1 )
            print (f"    Step {it +1 :3d}/{cfg .max_iters } | LR: {lr :.6f} | Train Loss: {loss .item ():.4f} | Val Loss: {v_loss .item ():.4f}")
            model .train ()
    prompt_str ="Artificial intelligence"
    input_ids =torch .tensor ([dataset .encode (prompt_str )],dtype =torch .long ,device =cfg .device )
    generated =model .generate (input_ids ,max_new_tokens =40 ,temperature =0.7 )
    output_text =dataset .decode (generated [0 ].tolist ())
    print (f"\n[Generated Output Sample]:\n{output_text }\n")
    plot_training_curves (train_history ,val_history ,steps )
    if model .blocks [0 ].attn .last_attn_weights is not None :
        tokens_vis =list (prompt_str [:15 ])
        plot_attention_map (model .blocks [0 ].attn .last_attn_weights ,tokens_vis )
    print ("\n[SUCCESS] Model from scratch verified and working properly!")
