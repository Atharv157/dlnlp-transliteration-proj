import gradio as gr
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import json

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TAG_LIST = ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
            'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']

async def predict_pos_tags(sentence):
    sentence = word_tokenize(sentence)
    base_prompt = f"""You are a POS tagger. For the given input sentence, output a JSON object in the form:
{{
  "predictions": ["TAG1", "TAG2", ..., "TAGN"]
}}

Constraints:
- Use only tags from this list: {TAG_LIST}
- Number of tags must exactly equal the number of tokens in the sentence.

Important: 
- For punctuation use "." instead of "PUNCT".

Sentence: {sentence}
"""
    try:
        resp =  await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": base_prompt}],
                    temperature=0,
                    timeout=60,
                    response_format={"type": "json_object"}  # enforce strict JSON
                )
        output = resp.choices[0].message.content
        try:
            data = json.loads(output)
            predictions = data.get("predictions", None)

            if not predictions or len(predictions) != len(sentence):
                return {"error": "Number of tokens does not match the length of predictions", "input_sentence": sentence, "predictions": predictions}
            
            return {"predictions": [[sentence[i], predictions[i]] for i in range(len(sentence))]}

        except Exception as e:
            return {"error": f"Couldn't parse JSON from response: {str(e)}"}

    except:
        return "Timeout error, please try again."



demo = gr.Interface(
    fn=predict_pos_tags,
    inputs=gr.Textbox(label="Enter a sentence"),
    outputs=gr.JSON(label="POS Tags"),
    title="GPT-4o-mini POS Tagger",
    description="Type a sentence and get POS tags predicted by the BiLSTM model."
)

if __name__ == "__main__":
    demo.launch()
