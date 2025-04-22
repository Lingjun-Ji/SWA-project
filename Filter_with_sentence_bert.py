import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer, util

# Load SDG 17 file
df_17 = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/SDG Question/sdg17_questions.csv")

# Load all content.csv files from output folder (absolute path)
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")
content_data = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.lower() == "content.csv":
            org_name = os.path.basename(root)
            path = os.path.join(root, file)
            try:
                cdf = pd.read_csv(path)
            except Exception:
                continue
            cdf["Organization"] = org_name
            content_data.append(cdf)

# Combine content data or create empty DataFrame
if content_data:
    all_content_df = pd.concat(content_data, ignore_index=True)
else:
    all_content_df = pd.DataFrame(columns=["Organization","URL","Raw Content","Page Count","File Type","Publication Date","Date Collected"])
all_content_df.columns = [c.strip() for c in all_content_df.columns]
all_content_df["Publication Date"] = pd.to_datetime(all_content_df["Publication Date"], errors='coerce')

# Build candidate sentences with metadata
records = []
for _, doc in all_content_df.iterrows():
    org = doc["Organization"]
    url = doc.get("URL", "")
    ptype = doc.get("File Type", "")
    pub_date = doc.get("Publication Date", "")
    last_date = doc.get("Date Collected", "")
    raw = str(doc.get("Raw Content", ""))
    page_blocks = re.split(r"===== PAGE (\d+) =====", raw)
    for i in range(1, len(page_blocks), 2):
        page = page_blocks[i]
        text = page_blocks[i+1] if i+1 < len(page_blocks) else ""
        for sent in re.split(r'(?<=[.!?]) +', text):
            sent = sent.strip()
            if sent:
                records.append({
                    "Organization": org,
                    "URL": url,
                    "Page": page,
                    "Document Type": ptype,
                    "Publication Date": pub_date,
                    "Last updated Date": last_date,
                    "Sentence": sent
                })

cand_df = pd.DataFrame(records)
if cand_df.empty:
    cand_df = pd.DataFrame(columns=["Organization","URL","Page","Document Type","Publication Date","Last updated Date","Sentence"])

# Initialize SBERT model and encode all candidate sentences
model = SentenceTransformer("all-MiniLM-L6-v2")
# Pre-compute embeddings for all sentences as a tensor
embeddings_tensor = model.encode(cand_df["Sentence"].tolist(), convert_to_tensor=True)

# Function to retrieve top-K semantically similar sentences per question

SIMILARITY_THRESHOLD = 0.4
TOP_K = 8

def get_top_sentences(question, org, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    # Encode the question
    q_emb = model.encode(question, convert_to_tensor=True)
    # Filter candidates by organization
    org_cands = cand_df[cand_df["Organization"] == org]
    if org_cands.empty:
        return [], [], [], [], [], []
    # Gather indices of those candidates
    indices = org_cands.index.tolist()
    # Select corresponding embeddings
    cand_embs = embeddings_tensor[indices]
    # Compute cosine similarities
    scores = util.cos_sim(q_emb, cand_embs)[0]
    top_indices = [(s.item(), i.item()) for s, i in zip(*scores.topk(k=len(scores))) if s >= threshold]
    if not top_indices and len(scores) > 0:
        max_score_idx = scores.argmax().item()
        top_indices = [(scores[max_score_idx].item(), max_score_idx)]
    else:
        top_indices = top_indices[:top_k]
    sentences, pages, urls, types, pubs, lasts = [], [], [], [], [], []
    for score, idx in top_indices:
        rec = org_cands.iloc[idx]
        sentences.append(f"[Page {rec['Page']}] {rec['Sentence']}")
        pages.append(str(rec["Page"]))
        urls.append(rec["URL"])
        types.append(rec["Document Type"])
        pubs.append(str(rec["Publication Date"]))
        lasts.append(str(rec["Last updated Date"]))
    return sentences, pages, urls, types, pubs, lasts

# Apply retrieval to each question
filtered, f_urls, f_pages, f_types, f_pubs, f_lasts = [], [], [], [], [], []
for _, row in df_17.iterrows():
    org = row["Organization"]
    q = row["SDG Question"]
    sents, pages, urls, types, pubs, lasts = get_top_sentences(q, org)
    filtered.append(" | ".join(sents))
    f_urls.append(" | ".join(set(urls)))
    f_pages.append(" | ".join(set(pages)))
    f_types.append(types[0] if types else "")
    f_pubs.append(pubs[0] if pubs else "")
    f_lasts.append(lasts[0] if lasts else "")

# Attach results to df_17
# (Remove leading spaces to fix indentation)
df_17["Filtered Content"] = filtered
df_17["URL"] = f_urls
df_17["Page number"] = f_pages
df_17["Document Type"] = f_types
df_17["Publication Date"] = f_pubs
df_17["Last updated Date"] = f_lasts

# Drop embedding column if present
if "emb" in df_17.columns:
    df_17 = df_17.drop(columns=["emb"])

# Save output CSV next to script
out_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(out_dir, "sdg17_questions_with_filtered_content.csv")
df_17.to_csv(output_file, index=False)
print(f"SDG17 filtered content saved to: {output_file}")
