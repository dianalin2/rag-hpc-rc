import requests
import time
import csv
import uuid
import os

API_URL = "http://localhost:5050"

OUTPUT_CSV = "rc_rag_benchmark.csv"

# -----------------------------
# Test Questions
# -----------------------------
TEST_QUERIES = [
  "What is Rivanna?",
  "How do I submit a Slurm job?",
  "How do I check GPU usage?",
  "What partitions are available on Rivanna?",
  "What is Open OnDemand?",
  "How do I install Python packages?",
  "How do I run Jupyter on Rivanna?",
  "What storage options do I have?",
  "How do I request a project allocation?",
  "What GPUs are available?",
  "What is the difference between CPU and GPU nodes?",
  "What is the policy on large jobs?",
  "What MPI versions exist on Rivanna?",
  "Why does my job stay pending?",
  "How do I transfer files?",
  "What is Slurm?",
  "How do I use modules?",
  "What tools are available for bioinformatics?",
  "What is AlphaFold?",
  "What versions of CUDA are available?"
]

# models to test
MODELS = [
"gemma3",
"mistral",
"llama3.2",
"phi3"
]


# -----------------------------
# Helper: Create chat
# -----------------------------
def create_chat():
    print("\nSending POST request to /create_chat...")
    print(f"➡️ URL: {API_URL}/create_chat")

    r = None
    try:
        r = requests.post(f"{API_URL}/create_chat")
        print(f"Response status code: {r.status_code}")
        print(f"Raw response text: {r.text}")
        r.raise_for_status()
    except Exception as e:
        print("ERROR during create_chat()")
        print("Exception:", e)
        if r is not None:
            print("Response:", r.text)
        raise

    chat_id = r.json()["chat_id"]
    print(f"Chat created with ID: {chat_id}")
    return chat_id

# -----------------------------
# Helper: Run a single query
# -----------------------------
def run_query(chat_id, query):
    print("\n========================================")
    print(f"Running query: {query}")
    print("========================================")

    payload = {
        "chat_id": chat_id,
        "query": query
    }

    print("\nSending POST request to /query...")
    print(f"➡️ URL: {API_URL}/query")
    print(f"➡️ Payload: {payload}")

    start = time.time()

    r = None
    try:
        r = requests.post(f"{API_URL}/query", json=payload)
    except Exception as e:
        print("Exception while sending HTTP request:", e)
        raise

    latency = time.time() - start
    print(f"Request finished in {latency:.2f} seconds")
    print(f"Response status code: {r.status_code}")
    print(f"Raw response text: {r.text}")

    try:
        data = r.json()
        print("Parsed JSON:", data)
    except Exception as e:
        print("ERROR parsing JSON response")
        print("Exception:", e)
        return {
            "error": "Invalid JSON response",
            "latency": latency,
            "answer": "",
            "sources": []
        }

    answer = data.get("response", "")
    sources = data.get("sources", [])

    print("Extracted fields:")
    print(f"  - Answer length: {len(answer)} characters")
    print(f"  - Sources count: {len(sources)}")

    return {
        "latency": latency,
        "answer": answer,
        "sources": sources
    }

# -----------------------------
# Main Benchmark
# -----------------------------

def main():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Query", "Latency", "HasSources", "Snippet"])

        for model in MODELS:
            print(f"\nBenchmarking model: {model}")

            os.environ["LLM_MODEL"] = model

            chat_id = create_chat()

            for q in TEST_QUERIES:
                print(f"→ {model} | {q}")
                result = run_query(chat_id, q)

                answer_snip = result["answer"][:150].replace("\n", " ")
                has_sources = len(result["sources"]) > 0

                writer.writerow([
                    model,
                    q,
                    f"{result['latency']:.2f}",
                    has_sources,
                    answer_snip
                ])


if __name__ == "__main__":
    main()
