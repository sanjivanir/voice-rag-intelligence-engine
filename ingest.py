import re

def get_msmarco_passages(limit=100):
    passages = [
        "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.",
        "The human heart is a muscular organ that pumps blood through the blood vessels of the circulatory system. Blood provides the body with oxygen and nutrients.",
        "Gravity is a fundamental interaction which causes mutual attraction between all things with mass or energy. It is by far the weakest of the four fundamental interactions.",
        "An electric current is a stream of charged particles, such as electrons or ions, moving through an electrical conductor or space.",
        "Internal combustion engines generate power by burning fuel inside a chamber to drive pistons or turbines.",
        "DNA is a molecule that carries most of the genetic instructions used in the development, functioning, and reproduction of all known organisms.",
        "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit under standard atmospheric pressure conditions at sea level.",
        "Computer processors execute instructions contained in computer programs to perform basic arithmetical, logical, control and input/output operations."
    ]
    passages = passages * (limit // len(passages) + 1)
    return passages[:limit]

# Strategy 1: Fixed-Size Character Overlap
def fixed_overlap_chunking(texts, chunk_size=120, overlap=30):
    chunks = []
    for text in texts:
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
    return chunks

# Strategy 2: Sentence / Structural Boundary Chunking
def sentence_chunking(texts, max_sentences=2):
    chunks = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i in range(0, len(sentences), max_sentences):
            chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks

# Strategy 3: Metadata-Aware Contextual Chunking
def metadata_aware_chunking(texts):
    chunks = []
    for idx, text in enumerate(texts):
        # Inject structural metadata tag into each chunk payload
        meta_chunk = f"[Document ID: doc_{idx:03d} | Domain: General Science] {text}"
        chunks.append(meta_chunk)
    return chunks