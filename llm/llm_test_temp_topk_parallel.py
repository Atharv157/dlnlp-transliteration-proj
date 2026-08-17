import os
import json
import hashlib
import asyncio
from ast import literal_eval
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI
import aiofiles
from time import sleep
from openai import APIError, RateLimitError, Timeout
import random

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 2
CONCURRENCY = 2  # number of parallel requests
WORDS_PER_PROMPT = 5  # number of words to process in each API call

results_lock = asyncio.Lock()
failed_lock = asyncio.Lock()

# Suggested values
TEMPERATURE_VALUES = [0.0, 0.4, 0.8, 1.2]
TOP_P_VALUES = [0.7, 0.85, 1.0]

async def append_jsonl(path, record, lock):
    async with lock:
        async with aiofiles.open(path, "a") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")

def hash_word(word):
    return hashlib.md5(" ".join(word).encode()).hexdigest()

def hash_word_list(word_list):
    """Create a hash for a list of words"""
    return hashlib.md5("|".join([" ".join(word) for word in word_list]).encode()).hexdigest()

async def call_llm_batch(words_batch, temperature=0.7, top_p=0.9, retries=MAX_RETRIES):
    """Process multiple words in a single API call"""
    words_str = ", ".join([f'"{word}"' for word in words_batch])
    
    base_prompt = f"""You are a Transliterator model Roman Hindi to Hindi. For the given input words, output a JSON object in the form:
{{
  "predictions": [
    {{"original": "<word1>", "transliterated": "<transliterated-word1>"}},
    {{"original": "<word2>", "transliterated": "<transliterated-word2>"}},
    ...
  ]
}}

Constraints:
- only give json output with format as mentioned above, no need for explanation
- maintain the same order as the input words
- output exactly {len(words_batch)} predictions

Roman Hindi words: [{words_str}]
"""
    prompt = base_prompt
    last_output = None
    backoff = 2  # initial wait time

    for attempt in range(1, retries + 1):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                timeout=60,
                response_format={"type": "json_object"}  # enforce strict JSON
            )
            output = resp.choices[0].message.content
            data = json.loads(output)

            predictions = data.get("predictions", [])
            last_output = predictions

            if (predictions and 
                isinstance(predictions, list) and 
                len(predictions) == len(words_batch) and
                all(isinstance(p, dict) and "transliterated" in p for p in predictions)):
                
                # Extract just the transliterated words in order
                return [p["transliterated"] for p in predictions]

        except RateLimitError:
            wait_time = backoff + random.uniform(0, 2)
            print(f"⚠️ Rate limit hit. Waiting {wait_time:.1f}s before retrying...")
            await asyncio.sleep(wait_time)
            backoff *= 2  # exponential backoff

        except (APIError, Timeout) as e:
            print(f"⚠️ API error: {e}. Retrying in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff *= 2

        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            if attempt == retries:
                return {"error": str(e), "raw_output": last_output}

    return last_output

def batch_words(english_words, batch_size=WORDS_PER_PROMPT):
    """Split words into batches of specified size"""
    return [english_words[i:i + batch_size] for i in range(0, len(english_words), batch_size)]

async def process_batch(semaphore, word_batch, completed_hashes, pbar, temperature, top_p, results_file, failed_file):
    batch_hash = hash_word_list(word_batch)
    
    # Check if any word in this batch is already processed
    unprocessed_words = []
    unprocessed_indices = []
    
    for i, word in enumerate(word_batch):
        word_hash = hash_word(word)
        if word_hash not in completed_hashes:
            unprocessed_words.append(word)
            unprocessed_indices.append(i)
    
    # If all words are already processed, just update progress bar
    if not unprocessed_words:
        pbar.update(len(word_batch))
        return
    
    # If only some words are processed, we'll process the unprocessed ones as a smaller batch
    words_to_process = unprocessed_words
    original_indices = unprocessed_indices
    
    async with semaphore:
        predictions = await call_llm_batch(words_to_process, temperature=temperature, top_p=top_p)
        
        if (isinstance(predictions, list) and 
            len(predictions) == len(words_to_process) and
            all(isinstance(p, str) and len(p) > 0 for p in predictions)):
            
            # ✅ valid batch -> save all results
            for i, (word, pred) in enumerate(zip(words_to_process, predictions)):
                record = {
                    "english word": word,
                    "native word": pred,
                    "temperature": temperature,
                    "top_p": top_p
                }
                await append_jsonl(results_file, record, results_lock)
        else:
            # ❌ invalid batch -> save raw wrong output to failed.jsonl for each word
            for word in words_to_process:
                record = {
                    "english word": word,
                    "native word": predictions,
                    "temperature": temperature,
                    "top_p": top_p
                }
                await append_jsonl(failed_file, record, failed_lock)
    
    pbar.update(len(word_batch))

def load_dataset(input_file):
    english_words = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                obj = json.loads(line)
                word = obj.get('english word', None)
                if word:
                    english_words.append(word)
            except:
                print(f"Skipping invalid line: {line}")
    return english_words

async def main(input_file):
    english_words = load_dataset(input_file)

    for temp in TEMPERATURE_VALUES:
        for p in TOP_P_VALUES:
            # Create unique filenames for each setting
            results_file = f"results_temp{temp}_topp{p}.jsonl"
            failed_file = f"failed_temp{temp}_topp{p}.jsonl"

            completed_hashes = set()
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            h = hash_word(data["english word"])
                            completed_hashes.add(h)
                        except:
                            pass

            # Create batches of words
            word_batches = batch_words(english_words, WORDS_PER_PROMPT)
            
            print(f"\nRunning for temperature={temp}, top_p={p}")
            print(f"Total words: {len(english_words)}")
            print(f"Total batches: {len(word_batches)}")
            print(f"Words per batch: {WORDS_PER_PROMPT}")

            semaphore = asyncio.Semaphore(CONCURRENCY)
            
            with tqdm(total=len(english_words), desc=f"Processing t={temp}, p={p}", unit="word") as pbar:
                tasks = [
                    process_batch(semaphore, batch, completed_hashes, pbar, temp, p, results_file, failed_file)
                    for batch in word_batches
                ]
                await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main("../data/raw/hin/hin_test.json"))