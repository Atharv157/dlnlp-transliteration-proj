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

# async def call_llm(word, temperature=0.7, top_p=0.9, retries=MAX_RETRIES):
#     base_prompt = f"""You are a Transliterator model Roman Hindi to Hindi. For the given input word, output a JSON object in the form:
# {{
#   "prediction": "<transliterated-word>"
# }}

# Constraints:
# - only give json output with format as mentioned above, no need for explanation

# Roman Hindi word: {word}
# """
#     prompt = base_prompt
#     last_output = None

#     for attempt in range(1, retries + 1):
#         try:
#             resp = await client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=temperature,
#                 top_p=top_p,
#                 timeout=60,
#                 response_format={"type": "json_object"}  # enforce strict JSON
#             )
#             output = resp.choices[0].message.content
#             data = json.loads(output)

#             pred = data.get("prediction", None)
#             last_output = pred

#             if pred:
#                 return pred

#             # Healing prompt if invalid
#             prompt = f"""Your previous output was: {output}

# This output was invalid because:
# - Either you gave something else than JSON or the prediction was empty
# Please try again.
# You are a Transliterator model Roman Hindi to Hindi. For the given input word, output a JSON object in the form:
# {{
# "prediction": "<transliterated-word>"
# }}

# Constraints:
# - only give json output with format as mentioned above, no need for explanation

# Roman Hindi word: {word}
# """
#         except Exception as e:
#             if attempt == retries:
#                 return {"error": str(e), "raw_output": last_output}
                

#     return last_output

async def call_llm(word, temperature=0.7, top_p=0.9, retries=MAX_RETRIES):
    base_prompt = f"""You are a Transliterator model Roman Hindi to Hindi. For the given input word, output a JSON object in the form:
{{
  "prediction": "<transliterated-word>"
}}

Constraints:
- only give json output with format as mentioned above, no need for explanation

Roman Hindi word: {word}
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
                response_format={"type": "json_object"},
            )

            output = resp.choices[0].message.content
            data = json.loads(output)
            pred = data.get("prediction", None)
            last_output = pred

            if pred:
                return pred

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



async def process_word(semaphore, word, completed_hashes, pbar, temperature, top_p, results_file, failed_file):
    word_hash = hash_word(word)
    if word_hash in completed_hashes:
        pbar.update(1)
        return

    async with semaphore:
        pred = await call_llm(word, temperature=temperature, top_p=top_p)
        record = {
            "english word": word,
            "native word": pred,
            "temperature": temperature,
            "top_p": top_p
        }

        if isinstance(pred, str) and len(pred):
            await append_jsonl(results_file, record, results_lock)
        else:
            await append_jsonl(failed_file, record, failed_lock)

    pbar.update(1)

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

            semaphore = asyncio.Semaphore(CONCURRENCY)
            print(f"\nRunning for temperature={temp}, top_p={p}")
            with tqdm(total=len(english_words), desc=f"Processing t={temp}, p={p}", unit="sent") as pbar:
                tasks = [
                    process_word(semaphore, word, completed_hashes, pbar, temp, p, results_file, failed_file)
                    for word in english_words
                ]
                await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main("../data/raw/hin/hin_test.json"))
