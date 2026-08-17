import gradio as gr
import torch
import json
import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv

# Add the current directory to path to import models
# sys.path.append(os.path.dirname(__file__))

# Import the existing models and functions
from src.models.lstm_seq2seq import initialize_lstm_model
from src.data.vocabulary import RomanVocabulary, DevanagariVocabulary
from train_transformer import TransformerSeq2Seq

# Load environment variables for LLM
load_dotenv()

##########################################################
# 🔹 LSTM Prediction (using existing functions)
##########################################################

def quick_load(checkpoint_path, device):
    """Quick load with weights_only=False - USE ONLY IF YOU TRUST THE SOURCE"""
    print("⚠️  Using unsafe loading - only use with trusted checkpoints!")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def transliterate_word_lstm(model, roman_word, roman_vocab, devanagari_vocab, device, max_length=50):
    """Transliterate a single word using the trained LSTM model - EXACT COPY from original"""
    model.eval()
    
    with torch.no_grad():
        # Encode input
        roman_chars = roman_vocab.split_roman_word(roman_word)
        roman_encoded = roman_vocab.encode(roman_chars)
        src_tensor = torch.tensor(roman_encoded).unsqueeze(0).to(device)
        
        # Encode source
        encoder_outputs, encoder_hidden, encoder_cell = model.encoder(src_tensor)
        
        # Initialize decoder
        if model.encoder.lstm.bidirectional:
            num_layers = model.encoder.num_layers
            hidden_dim = model.encoder.hidden_dim
            batch_size = 1
            
            encoder_hidden_reshaped = encoder_hidden.view(
                num_layers, 2, batch_size, hidden_dim
            )
            
            last_forward = encoder_hidden_reshaped[-1, 0]
            last_backward = encoder_hidden_reshaped[-1, 1]
            
            decoder_hidden_init = torch.cat([last_forward, last_backward], dim=1)
            decoder_hidden = model.init_hidden_proj(decoder_hidden_init)
            decoder_hidden = decoder_hidden.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        else:
            decoder_hidden = encoder_hidden[-1].unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        # Start with <sos> token
        decoder_input = torch.tensor([devanagari_vocab.char2idx['<sos>']]).to(device)
        
        decoded_indices = []
        
        for _ in range(max_length):
            output, decoder_hidden, decoder_cell, _ = model.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
            )
            
            # Get most likely character
            top1 = output.argmax(1)
            decoder_input = top1
            
            # Stop if EOS token
            if top1.item() == devanagari_vocab.char2idx['<eos>']:
                break
                
            decoded_indices.append(top1.item())
        
        # Convert indices to characters
        transliterated = devanagari_vocab.decode(decoded_indices)
        return transliterated

##########################################################
# 🔹 Transformer Prediction (using existing functions)
##########################################################

def make_src_key_padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (src == pad_idx)

def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask

def transliterate_word_transformer(model, roman_word, roman_vocab, devanagari_vocab, device, max_length=50):
    """Transliterate using Transformer with proper decoding - EXACT COPY from original"""
    model.eval()
    
    with torch.no_grad():
        # Prepare source
        roman_chars = roman_vocab.split_roman_word(roman_word)
        roman_encoded = roman_vocab.encode(roman_chars)
        src_tensor = torch.tensor(roman_encoded).unsqueeze(0).to(device)
        
        # Create source mask
        src_key_padding_mask = make_src_key_padding_mask(src_tensor, pad_idx=roman_vocab.char2idx.get("<pad>", 0))
        
        # Encode source
        memory = model.encode(src_tensor, src_key_padding_mask=src_key_padding_mask)
        
        # Start with SOS token
        sos_idx = devanagari_vocab.char2idx.get("<sos>", 1)
        decoder_input = torch.tensor([[sos_idx]], device=device)
        
        decoded_indices = []
        
        for step in range(max_length):
            # Create target mask and padding mask
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), device=device)
            tgt_key_padding_mask = make_src_key_padding_mask(decoder_input, pad_idx=devanagari_vocab.char2idx.get("<pad>", 0))
            
            # Transformer decoder
            output_step = model.decode(
                decoder_input,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
            )
            
            # Get prediction for the last token
            next_token_logits = output_step[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Stop if EOS token
            eos_idx = devanagari_vocab.char2idx.get("<eos>", 2)
            if next_token.item() == eos_idx:
                break
                
            decoded_indices.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Safety break if sequence gets too long
            if len(decoded_indices) >= max_length - 1:
                break
        
        # Convert indices to characters
        transliterated = devanagari_vocab.decode(decoded_indices)
        return transliterated

##########################################################
# 🔹 LLM Prediction (using existing functions)
##########################################################

from openai import AsyncOpenAI
import hashlib

async def call_llm(word, temperature=0.7, top_p=0.9):
    """Call LLM for transliteration - EXACT COPY from original"""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    base_prompt = f"""You are a Transliterator model Roman Hindi to Hindi. For the given input word, output a JSON object in the form:
{{
  "prediction": "<transliterated-word>"
}}

Constraints:
- only give json output with format as mentioned above, no need for explanation

Roman Hindi word: {word}
"""
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": base_prompt}],
            temperature=temperature,
            top_p=top_p,
            timeout=30,
            response_format={"type": "json_object"}
        )
        output = resp.choices[0].message.content
        data = json.loads(output)
        pred = data.get("prediction", "")
        return pred
    except Exception as e:
        return f"Error: {str(e)}"

def transliterate_word_llm(word):
    """Synchronous wrapper for LLM transliteration"""
    return asyncio.run(call_llm(word))

##########################################################
# 🔹 Model Loading (using existing initialization)
##########################################################

def load_lstm_model(checkpoint_path):
    """Load LSTM model using existing initialization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = quick_load(checkpoint_path, device)
    config = checkpoint.get('config', {})
    roman_vocab = checkpoint['roman_vocab']
    devanagari_vocab = checkpoint['devanagari_vocab']
    
    # Initialize model using existing function
    model = initialize_lstm_model(
        roman_vocab_size=len(roman_vocab),
        devanagari_vocab_size=len(devanagari_vocab),
        device=device,
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, roman_vocab, devanagari_vocab, device

def load_transformer_model(checkpoint_path):
    """Load Transformer model using existing initialization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = quick_load(checkpoint_path, device)
    config = checkpoint.get('config', {})
    roman_vocab = checkpoint['roman_vocab']
    devanagari_vocab = checkpoint['devanagari_vocab']
    
    # Get model parameters from config or use defaults
    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 8)
    num_encoder_layers = config.get("num_encoder_layers", 2)
    num_decoder_layers = config.get("num_decoder_layers", 2)
    dim_feedforward = config.get("dim_feedforward", 512)
    dropout = config.get("transformer_dropout", 0.1)
    max_len = config.get("max_len", 5000)
    
    pad_idx = roman_vocab.char2idx.get("<pad>", 0)
    
    # Initialize model using existing class
    model = TransformerSeq2Seq(
        src_vocab_size=len(roman_vocab),
        tgt_vocab_size=len(devanagari_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_idx=pad_idx,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, roman_vocab, devanagari_vocab, device

##########################################################
# 🔹 Main Gradio Interface
##########################################################

class TransliterationSystem:
    def __init__(self, lstm_checkpoint=None, transformer_checkpoint=None):
        self.lstm_checkpoint = lstm_checkpoint
        self.transformer_checkpoint = transformer_checkpoint
        
        self.lstm_model = None
        self.lstm_roman_vocab = None
        self.lstm_devanagari_vocab = None
        self.lstm_device = None
        
        self.transformer_model = None
        self.transformer_roman_vocab = None
        self.transformer_devanagari_vocab = None
        self.transformer_device = None
        
        self.models_loaded = False
    
    def load_models(self):
        """Load all models using existing initialization functions"""
        try:
            # Load LSTM model if checkpoint provided
            if self.lstm_checkpoint and os.path.exists(self.lstm_checkpoint):
                print(f"Loading LSTM model from: {self.lstm_checkpoint}")
                self.lstm_model, self.lstm_roman_vocab, self.lstm_devanagari_vocab, self.lstm_device = load_lstm_model(self.lstm_checkpoint)
                print("✅ LSTM model loaded")
            else:
                print("❌ LSTM checkpoint not provided or not found")
                self.lstm_model = None
            
            # Load Transformer model if checkpoint provided
            if self.transformer_checkpoint and os.path.exists(self.transformer_checkpoint):
                print(f"Loading Transformer model from: {self.transformer_checkpoint}")
                self.transformer_model, self.transformer_roman_vocab, self.transformer_devanagari_vocab, self.transformer_device = load_transformer_model(self.transformer_checkpoint)
                print("✅ Transformer model loaded")
            else:
                print("❌ Transformer checkpoint not provided or not found")
                self.transformer_model = None
            
            # Check if at least one model is loaded
            if self.lstm_model is not None or self.transformer_model is not None:
                self.models_loaded = True
                print("✅ Models loaded successfully!")
            else:
                print("⚠️  No models loaded. Only LLM will be available.")
                self.models_loaded = False
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def transliterate_sentence(self, sentence, model_type):
        """Transliterate a sentence (multiple words)"""
        if not sentence.strip():
            return ""
        
        words = sentence.strip().split()
        results = []
        
        for word in words:
            if model_type == "LSTM":
                if self.lstm_model is None:
                    results.append("❌ LSTM model not loaded")
                else:
                    result = transliterate_word_lstm(
                        self.lstm_model, word, self.lstm_roman_vocab, self.lstm_devanagari_vocab, self.lstm_device
                    )
                    results.append(result)
            elif model_type == "Transformer":
                if self.transformer_model is None:
                    results.append("❌ Transformer model not loaded")
                else:
                    result = transliterate_word_transformer(
                        self.transformer_model, word, self.transformer_roman_vocab, self.transformer_devanagari_vocab, self.transformer_device
                    )
                    results.append(result)
            elif model_type == "LLM":
                result = transliterate_word_llm(word)
                results.append(result)
            else:
                results.append(f"Unknown model: {model_type}")
        
        return " ".join(results)
    
    def predict(self, input_text, model_type):
        """Main prediction function"""
        if model_type != "LLM" and not self.models_loaded:
            return "❌ Models not loaded. Please check console for errors."
        
        if not input_text.strip():
            return "Please enter some text to transliterate."
        
        try:
            # Check if input is a single word or sentence
            if ' ' in input_text.strip():
                # It's a sentence
                return self.transliterate_sentence(input_text, model_type)
            else:
                # It's a single word
                if model_type == "LSTM":
                    if self.lstm_model is None:
                        return "❌ LSTM model not loaded. Please provide a valid checkpoint."
                    return transliterate_word_lstm(
                        self.lstm_model, input_text, self.lstm_roman_vocab, self.lstm_devanagari_vocab, self.lstm_device
                    )
                elif model_type == "Transformer":
                    if self.transformer_model is None:
                        return "❌ Transformer model not loaded. Please provide a valid checkpoint."
                    return transliterate_word_transformer(
                        self.transformer_model, input_text, self.transformer_roman_vocab, self.transformer_devanagari_vocab, self.transformer_device
                    )
                elif model_type == "LLM":
                    return transliterate_word_llm(input_text)
                else:
                    return f"Unknown model type: {model_type}"
                    
        except Exception as e:
            return f"❌ Error during transliteration: {str(e)}"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Roman Hindi to Hindi Transliteration System")
    parser.add_argument("--lstm-checkpoint", type=str, help="Path to LSTM model checkpoint")
    parser.add_argument("--transformer-checkpoint", type=str, help="Path to Transformer model checkpoint")
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio app on")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    
    return parser.parse_args()

def create_interface(transliterator):
    """Create Gradio interface"""
    with gr.Blocks(title="Roman Hindi to Hindi Transliteration", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🇮🇳 Roman Hindi to Hindi Transliteration System
            Compare different models for transliterating Roman Hindi (Latin script) to Hindi (Devanagari script)
            """
        )
        
        # Display loaded models status
        with gr.Row():
            status_markdown = gr.Markdown("")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="📝 Input Text (Roman Hindi)",
                    placeholder="Enter a word or sentence in Roman Hindi (e.g., 'namaste' or 'aap kaise hain')",
                    lines=2
                )
                
                # Dynamically set available models based on what's loaded
                available_models = []
                if transliterator.lstm_model is not None:
                    available_models.append("LSTM")
                if transliterator.transformer_model is not None:
                    available_models.append("Transformer")
                available_models.append("LLM")  # LLM is always available
                
                model_choice = gr.Radio(
                    choices=available_models,
                    label="🤖 Choose Model",
                    value=available_models[0] if available_models else "LLM",
                    info="Select which model to use for transliteration"
                )
                
                submit_btn = gr.Button("🚀 Transliterate", variant="primary")
                
                gr.Markdown(
                    """
                    ### ℹ️ About the Models:
                    - **LSTM**: Sequence-to-sequence model with attention mechanism
                    - **Transformer**: Transformer-based model with self-attention  
                    - **LLM**: GPT-4o-mini model for zero-shot transliteration
                    """
                )
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="🪷 Output Text (Hindi)",
                    placeholder="Transliterated text will appear here...",
                    lines=3
                )
                
                # gr.Markdown(
                #     """
                #     ### 💡 Examples to try:
                #     - Single words: `namaste`, `dhanyavad`, `bharat`
                #     - Sentences: `mera naam rahul hai`, `aaj mausam accha hai`
                #     - Phrases: `kya haal hai`, `shubh ratri`
                #     """
                # )
        
        # Update status message
        def update_status():
            status_lines = ["### 🔧 Loaded Models:"]
            if transliterator.lstm_model is not None:
                status_lines.append(f"- ✅ LSTM: {os.path.basename(transliterator.lstm_checkpoint)}")
            else:
                status_lines.append("- ❌ LSTM: Not loaded")
                
            if transliterator.transformer_model is not None:
                status_lines.append(f"- ✅ Transformer: {os.path.basename(transliterator.transformer_checkpoint)}")
            else:
                status_lines.append("- ❌ Transformer: Not loaded")
                
            status_lines.append("- ✅ LLM: Always available (requires OpenAI API key)")
            
            return "\n".join(status_lines)
        
        # Examples section
        # examples = [
        #     ["namaste", available_models[0] if available_models else "LLM"],
        #     ["dhanyavad", available_models[0] if available_models else "LLM"],
        # ]
        
        # if len(available_models) > 1:
        #     examples.append(["bharat", available_models[1]])
        
        # examples.extend([
        #     ["mera naam rahul hai", available_models[0] if available_models else "LLM"],
        #     ["aaj mausam accha hai", available_models[0] if available_models else "LLM"]
        # ])
        
        # gr.Examples(
        #     examples=examples,
        #     inputs=[input_text, model_choice],
        #     outputs=output_text,
        #     fn=transliterator.predict,
        #     cache_examples=False
        # )
        
        # Event handlers
        submit_btn.click(
            fn=transliterator.predict,
            inputs=[input_text, model_choice],
            outputs=output_text
        )
        
        # Also allow Enter key to submit
        input_text.submit(
            fn=transliterator.predict,
            inputs=[input_text, model_choice],
            outputs=output_text
        )
        
        # Set initial status
        demo.load(update_status, outputs=status_markdown)
    
    return demo

if __name__ == "__main__":
    args = parse_arguments()
    
    print("🚀 Starting Transliteration System...")
    print("📥 Loading models (this may take a moment)...")
    
    # Initialize transliteration system with checkpoint paths
    transliterator = TransliterationSystem(
        lstm_checkpoint=args.lstm_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint
    )
    
    # Load models
    transliterator.load_models()
    
    # Create and launch interface
    demo = create_interface(transliterator)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )