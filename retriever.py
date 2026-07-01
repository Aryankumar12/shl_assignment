import json
import re
import math

class BM25Retriever:
    def __init__(self, catalog_path: str):
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f, strict=False)
        
        # Prepare documents for indexing
        # Document representation will combine name, keys, and description
        self.documents = []
        for idx, item in enumerate(self.catalog):
            name = item.get("name", "")
            description = item.get("description", "")
            keys = " ".join(item.get("keys", []))
            job_levels = " ".join(item.get("job_levels", []))
            
            # Combine text fields with different weights
            # We can replicate text tokens to give them weight
            text = (
                (name + " ") * 5 + 
                (keys + " ") * 3 + 
                (job_levels + " ") * 2 + 
                description
            )
            self.documents.append({
                "index": idx,
                "item": item,
                "text": text
            })
            
        self.N = len(self.documents)
        self.k1 = 1.5
        self.b = 0.75
        
        # Tokenize and compute document lengths
        self.doc_tokens = []
        self.doc_lengths = []
        self.vocab = set()
        self.doc_freqs = {} # term -> document frequency
        
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.vocab.add(token)
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Compute IDFs
        self.idfs = {}
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF
            self.idfs[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            
    def _tokenize(self, text: str) -> list[str]:
        # Simple tokenizer: lowercase and split by word characters
        text = text.lower()
        # Replace non-alphanumeric with space
        text = re.sub(r'[^a-z0-9\s_+-]', ' ', text)
        return text.split()

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.catalog[:top_k]
            
        scores = []
        for idx in range(self.N):
            score = 0.0
            tokens = self.doc_tokens[idx]
            doc_len = self.doc_lengths[idx]
            
            # Count term frequencies in this document
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                
            for q_term in query_tokens:
                if q_term not in self.vocab:
                    continue
                q_tf = tf.get(q_term, 0)
                idf = self.idfs.get(q_term, 0.0)
                
                # BM25 formula
                numerator = q_tf * (self.k1 + 1)
                denominator = q_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator)
                
            scores.append((score, self.documents[idx]["item"]))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scores[:top_k] if score > 0.0]

    def match_assessment(self, query: str) -> dict or None:
        query_clean = query.strip().lower()
        if not query_clean:
            return None
            
        # 1. Exact match (case insensitive)
        for item in self.catalog:
            if item["name"].strip().lower() == query_clean:
                return item
                
        # 2. Check for substring match (prioritizing smaller name differences)
        best_match = None
        best_len_diff = 9999
        for item in self.catalog:
            name_clean = item["name"].strip().lower()
            if query_clean in name_clean or name_clean in query_clean:
                diff = abs(len(name_clean) - len(query_clean))
                if diff < best_len_diff:
                    best_len_diff = diff
                    best_match = item
        if best_match:
            return best_match

        # 3. Word overlap check
        query_words = set(self._tokenize(query))
        if not query_words:
            return None
            
        best_overlap = 0
        best_overlap_item = None
        for item in self.catalog:
            name_words = set(self._tokenize(item["name"]))
            overlap = len(query_words.intersection(name_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_item = item
                
        if best_overlap_item and best_overlap >= max(1, len(query_words) // 2):
            return best_overlap_item
            
        return None

# Simple mapping of key to test_type
KEY_TO_TEST_TYPE = {
    'Ability & Aptitude': 'A',
    'Biodata & Situational Judgment': 'B',
    'Competencies': 'C',
    'Development & 360': 'D',
    'Assessment Exercises': 'E',
    'Knowledge & Skills': 'K',
    'Personality & Behavior': 'P',
    'Simulations': 'S'
}

def get_test_type(item: dict) -> str:
    # If the item has multiple keys, map based on first key matching our KEY_TO_TEST_TYPE list
    keys = item.get("keys", [])
    for k in keys:
        if k in KEY_TO_TEST_TYPE:
            return KEY_TO_TEST_TYPE[k]
    # Default to K if no matching key is found
    return "K"
