import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import PreTrainedTokenizerFast
import streamlit as st

# Import your custom cleaning function directly
try:
    from Text_Cleaning import Sentiment_Text_Cleaning
except ImportError:
    st.error("Could not find Text_Cleaning.py in the project directory!")

# Automatically detect the directory where this app.py file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. ARCHITECTURE DEFINITION (Matches your original Set 4 exactly)
# ==============================================================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, rnn_type="lstm", dropout=0.0, pretrained_embeddings=None):
        super(SentimentRNN, self).__init__()        
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)        
        
        self.rnn_type = rnn_type.lower()
        rnn_dropout = dropout if n_layers > 1 else 0.0
        
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                               bidirectional=True, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, 
                              bidirectional=True, batch_first=True, dropout=rnn_dropout)            
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), 
            nn.Tanh(), 
            nn.Dropout(0.0),
            nn.Linear(hidden_dim * 2, 1)
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout) 
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text) 
        embedded = self.dropout(embedded)  
        output, _ = self.rnn(embedded) 
        output = self.dropout(output)
                       
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(output * attn_weights.expand_as(output), dim=1) 
        
        x = self.fc1(context)
        x = self.Tanh(x)
        x = self.dropout(x)
        logits = self.fc2(x) 
        
        return logits

# ==============================================================================
# 2. CACHED ASSET LOADING ENGINE
# ==============================================================================
@st.cache_resource
def load_prediction_assets(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Absolute paths targeting files sitting right next to app.py
    tokenizer_path = os.path.join(BASE_DIR, "saved_tokenizer")
    matrix_path = os.path.join(BASE_DIR, "embedding_matrix.pt")
    weight_file = os.path.join(BASE_DIR, "lstm_model.pt" if model_type == "LSTM" else "gru_model.pt")
    
    # 1. Load the exported tokenizer folder
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        raise FileNotFoundError(f"Tokenizer directory missing at: {tokenizer_path}")
        
    actual_vocab_size = len(tokenizer)
    
    # 2. Load the exact embedding matrix tensor used during training
    if os.path.exists(matrix_path):
        embedding_matrix = torch.load(matrix_path, map_location=device)
    else:
        raise FileNotFoundError(f"embedding_matrix.pt missing at: {matrix_path}")
    
    # 3. Instantiate model with identical hyperparameters from your Set 8
    model = SentimentRNN(
        vocab_size=actual_vocab_size,
        embedding_dim=300,
        hidden_dim=16,  
        output_dim=3,  
        n_layers=2,                     
        rnn_type=model_type.lower(),
        dropout=0.0,                    
        pretrained_embeddings=embedding_matrix,
    )
    
    # 4. Load the trained weights
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location=device))
    else:
        raise FileNotFoundError(f"Weight file missing at: {weight_file}")
        
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

# ==============================================================================
# 3. USER INTERFACE & LIVE INFERENCE
# ==============================================================================
st.set_page_config(page_title="Hinglish Sentiment Predictor", page_icon="🔮", layout="centered")
st.title("🔮 Hinglish Sentiment Prediction")
st.markdown("Evaluate Sentence Sentiments using LSTM & GRU Models.")

# Let user toggle between the model architectures smoothly
model_selection = st.selectbox("Select Model Architecture:", ["LSTM", "GRU"])

try:
    # Safely load the requested model assets
    model, tokenizer, device = load_prediction_assets(model_selection)
    st.caption(f"⚡ Loaded {model_selection} weights running on {str(device).upper()}")
    
    # Input field
    user_text = st.text_area("Type your Hinglish text here:")
    
    cleaned_text = Sentiment_Text_Cleaning(user_text)

    if st.button("Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please input some text to analyze.")
        else:
            # 1. RUN YOUR TEXT CLEANING PIPELINE PIPED FROM TEXT_CLEANING.PY
            cleaned_text = Sentiment_Text_Cleaning(user_text)
            cleaned_text = str(cleaned_text).strip()
            
            # Show the user what the text looks like after text cleanup rules run
            st.info(f"✨ Cleaned text used for model: *\"{cleaned_text}\"*")
            
            # 2. Encode inputs using your custom padding max length constraints (Set 1)
            encodings = tokenizer([cleaned_text], padding="max_length", truncation=True, max_length=42, return_tensors="pt")
            input_ids = encodings["input_ids"].to(device)
            
            # 3. Forward pass inference
            with torch.no_grad():
                logits = model(input_ids)
                probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                prediction = torch.argmax(logits, dim=1).item()
                
            # Class mapping configuration
            labels_map = {0: "🔴 Negative (0)", 1: "🟡 Neutral (1)", 2: "🟢 Positive (2)"}
            
            st.markdown("---")
            st.markdown(f"### Predicted Sentiment: **{labels_map[prediction]}**")
            
            # Show interactive breakdown chart
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Confidence Score': probabilities
            })
            st.bar_chart(prob_df.set_index('Sentiment'))
            
except Exception as e:
    st.error("⚠️ Setup Error: Could not locate required project files.")
    st.info(f"Make sure your export files match these paths inside: `{BASE_DIR}`")
    st.code(f"- {os.path.join(BASE_DIR, 'saved_tokenizer')}\n- {os.path.join(BASE_DIR, 'embedding_matrix.pt')}\n- {os.path.join(BASE_DIR, 'lstm_model.pt')}\n- {os.path.join(BASE_DIR, 'gru_model.pt')}")
    st.exception(e)