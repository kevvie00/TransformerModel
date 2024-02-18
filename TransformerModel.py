import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import math
import unicodedata
import re

#hyperparameters
batch = 32
context = 20
layers = 8
embedding_dim = 300
dK = 64
dV = 64
blocks = 6 #Nx in paper
loss_iters = 50
dropout_rate = 0.3
max_length = 20

#Vars
SOS = "<SOS>"
EOS = "<EOS>"
padding_token = 0
SOS_token = 1
EOS_token = 2

english_stoi = {SOS: 1, EOS: 2}
english_itos = {1: SOS, 2: EOS}
french_stoi = {SOS: 1, EOS: 2}
french_itos = {1: SOS, 2: EOS}


#--------Functions for pre-processing the data.

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s

#Add sentence to stoi and itos dictionaries for tokenization
def addSentenceToDicts(stoi, itos, sentence):
    for w in sentence:
        if w not in stoi:
            stoi[w] = len(stoi) + 1
            itos[len(stoi)] = w
    return

#--------Functions for training/evaluating the model.

#Gets a batch of english-french example pairs.
def get_batch(mode):
    data = training_data if mode =='train' else test_data
    starting_inds = [random.randint(0, len(data) - 1) for _ in range(batch)]
    eng = torch.stack([data[i_start][0] for i_start in starting_inds], dim=0).to(torch.long)
    fr = torch.stack([data[i_start][1] for i_start in starting_inds], dim=0).to(torch.long)
    return eng, fr

#Estimates current loss of the model.
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    training_loss, test_loss = 0.0, 0.0
    for i in range(loss_iters):
        training_data, targets = get_batch('train')
        logits, loss = model(training_data, targets)
        training_loss += loss.item()

        test_data, targets = get_batch('test')
        logits, loss = model(test_data, targets)
        test_loss += loss.item()
    training_loss /= loss_iters
    test_loss /= loss_iters
    model.train()
    return training_loss, test_loss

#Decodes tokenized french example back to French.
def decodeFrench(tensor):
    str = ""
    for i in range(len(tensor)):
        if(tensor[i].item() == padding_token): 
            break
        str += french_itos[tensor[i].item()] + ' ';
    return str;

#Decodes tokenized english example back to English.
def decodeEnglish(tensor):
    str = ""
    for i in range(len(tensor)):
        if(tensor[i].item() == padding_token): 
            break
        str += english_itos[tensor[i].item()] + ' ';
    return str;

# Specify the file path
file_path = 'french-eng.txt'
# Read the file and split into lines
lines = open(file_path, 'r', encoding='utf-8').read().strip().split('\n')
# Split every line into pairs and normalize
pairs = [[normalizeString(s) for s in l.split('\t')][:-1] for l in lines]

data = []
for pair in pairs:
    english = pair[0].split()
    french = pair[1].split()
    eng = [SOS_token]
    fr = [SOS_token]

    addSentenceToDicts(english_stoi, english_itos, english)
    addSentenceToDicts(french_stoi, french_itos, french)
    for w in english:
        eng.append(english_stoi[w])
    for w in french:
        fr.append(french_stoi[w])

    #manually add EOS token
    eng.append(EOS_token)
    fr.append(EOS_token)

    #force all data examples to be the same size
    eng += [padding_token] * (max_length - len(eng))
    fr += [padding_token] * (max_length - len(fr))

    eng = torch.tensor(eng, dtype=torch.int32)
    fr = torch.tensor(fr, dtype=torch.int32)
    if len(eng) <= max_length and len(fr) <= max_length:  #originally have 228556 examples, we only lose ~2000 examples.
        data.append((eng, fr))

assert (len(english_stoi) == len(english_itos)) and (len(french_stoi) == len(french_itos)),"Ensure tokenization dictioanries are the same size."
src_vocab_size = len(english_stoi)
trg_vocab_size = len(french_stoi)

#Split data
random.shuffle(data)
train_size = int(0.8 * len(data))
training_data = data[:train_size]
test_data = data[train_size:]

#Model classes
class SingleHeadAttention(nn.Module):
    def __init__(self, embed_size, dK, dV):
        super().__init__()
        self.query = nn.Linear(embed_size, dK, bias=False)
        self.key = nn.Linear(embed_size, dK, bias=False)
        self.value = nn.Linear(embed_size, dV, bias=False)
        self.dK = dK
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, padding_mask=None, decoder_mask=False):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)   #B,T,dV
        similarity_score = torch.matmul(Q, K.permute(0,2,1)) #B, T, T
        if decoder_mask:
            similarity_score = torch.where(torch.tril(similarity_score) == 0.0, float('-inf'), similarity_score)
        if padding_mask is not None: 
             similarity_score = similarity_score.masked_fill(~padding_mask, float('-inf'))
        similarity_score /= math.sqrt(self.dK)
        affinity = F.softmax(similarity_score, dim=-1)
        affinity = self.dropout(affinity)
        attention_score = torch.matmul(affinity, V) #B,T,dV
        return attention_score
    
class MultiHeadAttention(nn.Module):
    def __init__(self, layers, embed_size, dK, dV):
        assert dK%layers==0 and dV%layers==0, "Key and value dimensions must be divisible by layers."
        super().__init__()
        self.heads = [SingleHeadAttention(embed_size, dK//layers, dV//layers) for _ in range(layers)]
        self.linear = nn.Linear(dV, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, query, key, value, padding_mask=None, decoder_mask=False):
        head_outputs = [head(query, key, value, padding_mask, decoder_mask) for head in self.heads]
        output = torch.cat(head_outputs, dim=-1)
        output = self.linear(output) #Last dimension is embed_size so we can use residuals later.
        output = self.dropout(output)
        return output
            
class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, 4 * embed_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, layers, embedding_dim, dK, dV):
        super().__init__()
        self.maskedhead = MultiHeadAttention(layers, embedding_dim, dK, dV)
        self.crosshead = MultiHeadAttention(layers, embedding_dim, dK, dV)
        self.feedforward = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ln3 = nn.LayerNorm(embedding_dim)
    def forward(self, x, enc_key, enc_value, decoder_mask=True):
        x_norm = self.ln1(x)  #pre-norm
        x = x + self.maskedhead(x_norm, x_norm, x_norm, None, decoder_mask) #shape (B,T,C)
        x = x + self.crosshead(self.ln2(x), self.ln2(enc_key), self.ln2(enc_value))
        x = x + self.feedforward(self.ln3(x)) #shape (B,T,C)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, layers, embedding_dim, dK, dV):
        super().__init__()
        self.multihead = MultiHeadAttention(layers, embedding_dim, dK, dV)
        self.feedforward = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, padding_mask=None):
        x_norm = self.ln1(x) #pre-norm
        x = x + self.multihead(x_norm, x_norm, x_norm, padding_mask) #shape (B,T,C)
        x = x + self.feedforward(self.ln2(x)) #shape (B,T,C)
        return x
    
class Transformer(nn.Module):
    def __init__(self, layers, embedding_dim, dK, dV, src_vocab_size, trg_vocab_size, blocks):
        super().__init__()
        self.src_tok_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.trg_tok_embedding = nn.Embedding(trg_vocab_size, embedding_dim)
        self.src_pos_embedding = nn.Embedding(context, embedding_dim)
        self.trg_pos_embedding = nn.Embedding(context, embedding_dim)
        self.encoder = nn.ModuleList([EncoderBlock(layers, embedding_dim, dK, dV) for _ in range(blocks)])
        self.decoder = nn.ModuleList([DecoderBlock(layers, embedding_dim, dK, dV) for _ in range(blocks)])
        
        self.linear = nn.Linear(embedding_dim, trg_vocab_size) #for decoder
        
    def forward(self, src, trg):
        mask = self.create_mask(src)
        src = self.src_tok_embedding(src) #shape (B,T,C)

        trg_embed = trg[:,:-1]
        trg_embed = self.trg_tok_embedding(trg_embed) #shape (B,T,C)  assuming src and trg is of shape (B,T)

        src = src + self.src_pos_embedding(torch.arange(src.size(1)))
        trg_embed = trg_embed + self.trg_pos_embedding(torch.arange(trg_embed.size(1)))

        enc_output = src
        for encoder in self.encoder:
            enc_output = encoder(enc_output, mask)

        output = trg_embed
        for decoder in self.decoder:
            output = decoder(output, enc_output, enc_output)
        output = self.linear(output)
        loss = F.cross_entropy(output.permute(0, 2, 1), trg[:,1:], ignore_index=padding_token)
        return output, loss

    def create_mask(self, input_sequence):
        # Create a mask for padded positions
        mask = (input_sequence != 0).unsqueeze(1)
        return mask

    def generate_french_sequence(self, src, max_len=max_length):
        mask = self.create_mask(src)
    
        src = self.src_tok_embedding(src)
        src = src + self.src_pos_embedding(torch.arange(src.size(1), device=src.device))
        
        enc_output = src
        for encoder in self.encoder:
            enc_output = encoder(enc_output, mask)
    
        # Initialize the target sequence with the start token
        trg = torch.full((batch, 1), SOS_token, device=src.device)
        # Calculate positional embeddings for the entire sequence once
        trg_pos = self.trg_pos_embedding(torch.arange(max_len, device=trg.device))
        
        for _ in range(max_len):
            trg_embedding = self.trg_tok_embedding(trg)
            trg_embedding = trg_embedding + trg_pos[_]  # Use the corresponding positional embedding
    
            decoder_output = trg_embedding  # Initialize with token embedding
            for decoder_block in self.decoder:
                decoder_output = decoder_block(decoder_output, enc_output, enc_output, False)
    
            # Get the last token from the decoder output
            last_token_output = decoder_output[:, -1, :].unsqueeze(1)
    
            # Predict the next token
            next_token_probs = F.softmax(self.linear(last_token_output), dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1)
            # Append the predicted token to the target sequence
            trg = torch.cat([trg, next_token], dim=1)
    
            # Check if the generated sequence ends with the end token
            if next_token[0].item() == EOS_token:
                break
    
        return trg.squeeze(0)
    
model = Transformer(layers, embedding_dim, dK, dV, src_vocab_size, trg_vocab_size, blocks)

#Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

# Training loop
steps = 10000

for step in range(steps):
    training_batch, targets = get_batch('train')
    logits, loss = model(training_batch, targets)
    next_token = torch.argmax(logits, dim=-1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Update model parameters
    optimizer.step()
    scheduler.step()
        
    
    # Print loss for monitoring training progress
    if step!=0 and step%100 == 0:
        training_loss, test_loss = estimate_loss(model)
        print(f"Step {step}/{steps}, Train Loss: {training_loss}, Test Loss: {test_loss}")



#Testing
        
model.eval()  
english_sequence, _ = get_batch('test')
generated_french_sequence = model.generate_french_sequence(english_sequence)

for i in range(batch):
    print(decodeEnglish(english_sequence[i]))
    print(decodeFrench(generated_french_sequence[i]))