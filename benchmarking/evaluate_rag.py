"""
Script to evaluate RAG app responses for perplexity and hallucination.

This script:
1. Queries the RAG app with test questions
2. Calculates perplexity scores for generated responses
3. Detects potential hallucinations by comparing responses to source context
4. Outputs comprehensive evaluation results

Usage:
    # Make sure your RAG backend is running (default: http://localhost:5050)
    python evaluate_rag.py
    
    # Or set custom API URL:
    API_URL=http://localhost:5000 python evaluate_rag.py
    
    # Set the model to test:
    LLM_MODEL=gemma3 python evaluate_rag.py

Output:
    - rag_evaluation_results.csv: CSV file with all metrics
    - rag_evaluation_results.json: JSON file with full results

Dependencies:
    Required:
    - requests (for API calls)
    
    Optional (for full functionality):
    - transformers, torch (for perplexity calculation)
    - numpy, scikit-learn (for statistics and semantic similarity)
    - langchain-ollama (for embedding-based hallucination detection)
"""

import requests
import time
import csv
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path to import perplexity module
sys.path.append(str(Path(__file__).parent))
try:
    from perplexity import calculate_perplexity
    HAS_PERPLEXITY = True
except ImportError as e:
    HAS_PERPLEXITY = False
    print(f"Warning: Could not import perplexity module: {e}")
    print("Perplexity calculations will be skipped. Install transformers and torch to enable.")
    def calculate_perplexity(text: str) -> Optional[float]:
        return None

# For hallucination detection using embeddings
try:
    from langchain_community.embeddings import OllamaEmbeddings

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_EMBEDDINGS = True
    HAS_NUMPY = True
except ImportError:
    HAS_EMBEDDINGS = False
    HAS_NUMPY = False
    print("Warning: langchain_ollama or sklearn not available. Hallucination detection will be limited.")
    # Fallback for numpy operations
    try:
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False
        print("Warning: numpy not available. Summary statistics will be limited.")

API_URL = os.getenv("API_URL", "http://localhost:5050")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OUTPUT_CSV = "rag_evaluation_results.csv"
OUTPUT_JSON = "rag_evaluation_results.json"

# Test queries for evaluation
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

# Models to test (if you want to test multiple models)
MODELS = [
    os.getenv("LLM_MODEL", "gemma3")
]


def create_chat() -> str:
    """Create a new chat session with the RAG app."""
    try:
        r = requests.post(f"{API_URL}/create_chat", timeout=30)
        r.raise_for_status()
        chat_id = r.json()["chat_id"]
        return chat_id
    except Exception as e:
        print(f"ERROR during create_chat(): {e}")
        raise


def run_query(chat_id: str, query: str) -> Dict:
    """Run a query against the RAG app and return the response."""
    payload = {
        "chat_id": chat_id,
        "query": query
    }

    start = time.time()
    try:
        r = requests.post(f"{API_URL}/query", json=payload, timeout=120)
        r.raise_for_status()
        latency = time.time() - start
        
        data = r.json()
        return {
            "latency": latency,
            "response": data.get("response", ""),
            "sources": data.get("sources", []),
            "success": True
        }
    except Exception as e:
        print(f"ERROR during query: {e}")
        return {
            "latency": time.time() - start,
            "response": "",
            "sources": [],
            "success": False,
            "error": str(e)
        }


def calculate_semantic_similarity(text1: str, text2: str, embeddings_model) -> float:
    """Calculate cosine similarity between two texts using embeddings."""
    if not HAS_EMBEDDINGS:
        return 0.0
    
    try:
        emb1 = embeddings_model.embed_query(text1)
        emb2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Warning: Could not calculate semantic similarity: {e}")
        return 0.0


def detect_hallucination_simple(response: str, sources: List[str]) -> Dict:
    """
    Simple hallucination detection based on:
    1. Semantic similarity between response and sources
    2. Response length and structure
    3. Presence of uncertainty phrases
    """
    results = {
        "has_sources": len(sources) > 0,
        "response_length": len(response),
        "semantic_similarity": 0.0,
        "uncertainty_phrases": [],
        "hallucination_score": 0.0  # 0 = no hallucination, 1 = likely hallucination
    }
    
    # Check for uncertainty phrases (good sign - model admits uncertainty)
    uncertainty_phrases = [
        "i'm not sure",
        "i don't know",
        "based on the provided information",
        "not covered in the context",
        "unclear",
        "may vary"
    ]
    found_phrases = [phrase for phrase in uncertainty_phrases if phrase.lower() in response.lower()]
    results["uncertainty_phrases"] = found_phrases
    
    # If no sources, likely hallucination
    if not results["has_sources"]:
        results["hallucination_score"] = 0.8
        return results
    
    # Calculate semantic similarity if embeddings available
    if HAS_EMBEDDINGS:
        try:
            embeddings = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model="nomic-embed-text"
            )
            # Combine all sources into one text for comparison
            sources_text = " ".join(sources)
            similarity = calculate_semantic_similarity(response, sources_text, embeddings)
            results["semantic_similarity"] = similarity
            
            # Lower similarity = higher hallucination risk
            if similarity < 0.3:
                results["hallucination_score"] = 0.7
            elif similarity < 0.5:
                results["hallucination_score"] = 0.4
            elif similarity < 0.7:
                results["hallucination_score"] = 0.2
            else:
                results["hallucination_score"] = 0.1
        except Exception as e:
            print(f"Warning: Could not compute semantic similarity: {e}")
            results["hallucination_score"] = 0.5  # Unknown
    
    # Adjust score based on uncertainty phrases (lower score = less hallucination)
    if found_phrases:
        results["hallucination_score"] *= 0.7  # Reduce hallucination score if model shows uncertainty
    
    return results


def evaluate_response(query: str, response_data: Dict) -> Dict:
    """
    Comprehensive evaluation of a RAG response.
    Returns metrics including perplexity and hallucination detection.
    """
    response = response_data.get("response", "")
    
    if not response or not response_data.get("success", False):
        return {
            "query": query,
            "perplexity": None,
            "hallucination": None,
            "error": response_data.get("error", "No response received"),
            "latency": response_data.get("latency", 0),
            "has_sources": False,
            "response_length": 0
        }
    
    # Calculate perplexity
    perplexity = None
    if HAS_PERPLEXITY:
        try:
            perplexity = calculate_perplexity(response)
        except Exception as e:
            print(f"Warning: Could not calculate perplexity: {e}")
            perplexity = None
    
    # Detect hallucinations
    hallucination_results = detect_hallucination_simple(
        response, 
        response_data.get("sources", [])
    )
    
    return {
        "query": query,
        "response": response[:500],  # Truncate for storage
        "perplexity": perplexity,
        "hallucination_score": hallucination_results["hallucination_score"],
        "semantic_similarity": hallucination_results["semantic_similarity"],
        "has_sources": hallucination_results["has_sources"],
        "num_sources": len(response_data.get("sources", [])),
        "uncertainty_phrases": len(hallucination_results["uncertainty_phrases"]),
        "response_length": hallucination_results["response_length"],
        "latency": response_data.get("latency", 0),
        "sources": response_data.get("sources", [])
    }


def main():
    """Main evaluation loop."""
    print("=" * 60)
    print("RAG Evaluation: Perplexity & Hallucination Detection")
    print("=" * 60)
    
    all_results = []
    
    # Create CSV file
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model", "query", "perplexity", "hallucination_score", 
            "semantic_similarity", "has_sources", "num_sources",
            "uncertainty_phrases", "response_length", "latency", "response_snippet"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for model in MODELS:
            print(f"\nEvaluating model: {model}")
            os.environ["LLM_MODEL"] = model
            
            try:
                chat_id = create_chat()
                print(f"Chat created: {chat_id}")
            except Exception as e:
                print(f"Failed to create chat: {e}")
                continue
            
            for i, query in enumerate(TEST_QUERIES, 1):
                print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query}")
                
                # Run query
                response_data = run_query(chat_id, query)
                
                # Evaluate response
                evaluation = evaluate_response(query, response_data)
                evaluation["model"] = model
                
                all_results.append(evaluation)
                
                # Write to CSV
                writer.writerow({
                    "model": model,
                    "query": query,
                    "perplexity": f"{evaluation['perplexity']:.2f}" if evaluation['perplexity'] else "N/A",
                    "hallucination_score": f"{evaluation['hallucination_score']:.3f}" if evaluation.get('hallucination_score') is not None else "N/A",
                    "semantic_similarity": f"{evaluation['semantic_similarity']:.3f}" if evaluation.get('semantic_similarity') is not None else "N/A",
                    "has_sources": evaluation.get("has_sources", False),
                    "num_sources": evaluation.get("num_sources", 0),
                    "uncertainty_phrases": evaluation.get("uncertainty_phrases", 0),
                    "response_length": evaluation.get("response_length", 0),
                    "latency": f"{evaluation['latency']:.2f}",
                    "response_snippet": evaluation.get("response", "")[:200]
                })
                
                # Print summary
                if evaluation.get("perplexity"):
                    print(f"Perplexity: {evaluation['perplexity']:.2f}")
                if evaluation.get("hallucination_score") is not None:
                    print(f"Hallucination Score: {evaluation['hallucination_score']:.3f} (lower is better)")
                if evaluation.get("semantic_similarity") is not None:
                    print(f"Semantic Similarity: {evaluation['semantic_similarity']:.3f}")
                print(f"Sources: {evaluation.get('num_sources', 0)}")
                print(f"Latency: {evaluation['latency']:.2f}s")
    
    # Save full results as JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as jsonfile:
        json.dump(all_results, jsonfile, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    valid_perplexities = [r["perplexity"] for r in all_results if r.get("perplexity") is not None]
    valid_hallucination_scores = [r["hallucination_score"] for r in all_results if r.get("hallucination_score") is not None]
    valid_similarities = [r["semantic_similarity"] for r in all_results if r.get("semantic_similarity") is not None]
    
    if valid_perplexities:
        if HAS_NUMPY:
            print(f"Perplexity - Mean: {np.mean(valid_perplexities):.2f}, Median: {np.median(valid_perplexities):.2f}")
            print(f"Perplexity - Min: {np.min(valid_perplexities):.2f}, Max: {np.max(valid_perplexities):.2f}")
        else:
            mean_p = sum(valid_perplexities) / len(valid_perplexities)
            sorted_p = sorted(valid_perplexities)
            median_p = sorted_p[len(sorted_p) // 2]
            print(f"Perplexity - Mean: {mean_p:.2f}, Median: {median_p:.2f}")
            print(f"Perplexity - Min: {min(valid_perplexities):.2f}, Max: {max(valid_perplexities):.2f}")
    
    if valid_hallucination_scores:
        if HAS_NUMPY:
            print(f"Hallucination Score - Mean: {np.mean(valid_hallucination_scores):.3f}, Median: {np.median(valid_hallucination_scores):.3f}")
            print(f"Hallucination Score - Min: {np.min(valid_hallucination_scores):.3f}, Max: {np.max(valid_hallucination_scores):.3f}")
        else:
            mean_h = sum(valid_hallucination_scores) / len(valid_hallucination_scores)
            sorted_h = sorted(valid_hallucination_scores)
            median_h = sorted_h[len(sorted_h) // 2]
            print(f"Hallucination Score - Mean: {mean_h:.3f}, Median: {median_h:.3f}")
            print(f"Hallucination Score - Min: {min(valid_hallucination_scores):.3f}, Max: {max(valid_hallucination_scores):.3f}")
    
    if valid_similarities:
        if HAS_NUMPY:
            print(f"Semantic Similarity - Mean: {np.mean(valid_similarities):.3f}, Median: {np.median(valid_similarities):.3f}")
        else:
            mean_s = sum(valid_similarities) / len(valid_similarities)
            sorted_s = sorted(valid_similarities)
            median_s = sorted_s[len(sorted_s) // 2]
            print(f"Semantic Similarity - Mean: {mean_s:.3f}, Median: {median_s:.3f}")
    
    responses_with_sources = sum(1 for r in all_results if r.get("has_sources", False))
    print(f"Responses with sources: {responses_with_sources}/{len(all_results)}")
    
    print(f"\nResults saved to:")
    print(f"   - CSV: {OUTPUT_CSV}")
    print(f"   - JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

