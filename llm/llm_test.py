import os
import json
import hashlib
import asyncio
from ast import literal_eval
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI
import aiofiles

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 3
CONCURRENCY = 5  # number of parallel requests

results_lock = asyncio.Lock()
failed_lock = asyncio.Lock()

RESULTS_FILE = "results.jsonl"
FAILED_FILE = "failed.jsonl"

async def append_jsonl(path, record, lock):
    async with lock:
        async with aiofiles.open(path, "a") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # await f.write(json.dumps(record) + "\n")


def hash_word(word):
    return hashlib.md5(" ".join(word).encode()).hexdigest()


async def call_llm(word, retries=MAX_RETRIES):
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

    for attempt in range(1, retries + 1):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=60,
                response_format={"type": "json_object"}  # enforce strict JSON
            )
            output = resp.choices[0].message.content
            data = json.loads(output)

            pred = data.get("prediction", None)
            last_output = pred

            if (pred):
                return pred  # ✅ valid

            # healing prompt
            prompt = f"""Your previous output was: {output}

            This output was invalid because:
            - Either you gave something else than JSON or the prediction was empty
            Please try again.
            You are a Transliterator model Roman Hindi to Hindi. For the given input word, output a JSON object in the form:
            {{
            "prediction": "<transliterated-word>"
            }}

            Constraints:
            - only give json output with format as mentioned above, no need for explanation

            Roman Hindi word: {word}
            """
        except Exception as e:
            if attempt == retries:
                return {"error": str(e), "raw_output": last_output}

    # ❌ return last seen (invalid) preds
    return last_output


async def process_word(semaphore, word, completed_hashes, pbar):
    word_hash = hash_word(word)
    if word_hash in completed_hashes:
        pbar.update(1)
        return

    async with semaphore:
        pred = await call_llm(word)
        record = {"english word": word, "native word": pred}

        if (isinstance(pred, str) and
            len(pred)):
            # ✅ valid -> save to results.jsonl
            await append_jsonl(RESULTS_FILE, record, results_lock)
        else:
            # ❌ invalid -> save raw wrong output to failed.jsonl
            await append_jsonl(FAILED_FILE, record, failed_lock)

    pbar.update(1)

def load_dataset(input_file):
    english_words = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        try:
            for line in lines:
                object = json.loads(line)
                word = object.get('english word', None)
                if word:
                    english_words.append(word)

        except:
            print(f"Dataset loading function failed for the line: {line}")
    return english_words

async def main(input_file):
    english_words = load_dataset(input_file)

    completed_hashes = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    h = hash_word(data["english word"])
                    completed_hashes.add(h)
                except:
                    pass

    semaphore = asyncio.Semaphore(CONCURRENCY)

    with tqdm(total=len(english_words), desc="Processing", unit="sent") as pbar:
        tasks = [
            process_word(semaphore, word, completed_hashes, pbar)
            for word in english_words
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main("../data/raw/hin/hin_test.json"))
