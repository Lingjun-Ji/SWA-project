import pandas as pd
import re
import os
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load SDG 17 file
df_17 = pd.read_csv("C:/Users/KrisJ/Desktop/SWA_CODE/SDG Question/sdg17_questions.csv")

# Load content.csv files from output folder
content_data = []
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.lower() == "content.csv":
            org_name = os.path.basename(root)
            path = os.path.join(root, file)
            df = pd.read_csv(path)
            df["Organization"] = org_name
            content_data.append(df)

all_content_df = pd.concat(content_data, ignore_index=True)
all_content_df.columns = [c.strip() for c in all_content_df.columns]
all_content_df["Publication Date"] = pd.to_datetime(all_content_df["Publication Date"], errors='coerce')

# Extract keywords using KeyBERT
df_17 = df_17.copy()
kw_model = KeyBERT()

def extract_keywords(text):
    keywords = [kw[0] for kw in kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=list(ENGLISH_STOP_WORDS),
        use_mmr=True,
        diversity=0.5,
        top_n=6
    )]
    return keywords

df_17["Keywords"] = df_17["SDG Question"].apply(extract_keywords)

# Filter content
filtered_snippets = []

for _, row in df_17.iterrows():
    org = row["Organization"]
    keywords = row["Keywords"]

    org_contents = all_content_df[all_content_df["Organization"] == org].copy()
    org_contents = org_contents.dropna(subset=["Publication Date"])

    if org_contents.empty:
        filtered_snippets.append(("", "", "", "", "", ""))
        continue

    latest_docs = org_contents

    matched_sentences = []
    matched_pages = []
    urls = set()
    doc_type, pub_date, last_date = "", "", ""

    for _, doc in latest_docs.iterrows():
        content = str(doc.get("Raw Content", ""))
        page_blocks = re.split(r"===== PAGE (\d+) =====", content)
        for i in range(1, len(page_blocks), 2):
            page_num = page_blocks[i]
            page_text = page_blocks[i + 1] if (i + 1) < len(page_blocks) else ""
            for sent in re.split(r'(?<=[.!?]) +', page_text):
                if any(k.lower() in sent.lower() for k in keywords):
                    matched_sentences.append(sent.strip())
                    matched_pages.append(page_num.strip())
                    urls.add(doc.get("URL", ""))
                    if not doc_type:
                        doc_type = doc.get("File Type", "")
                    if not pub_date:
                        pub_date = str(doc.get("Publication Date", ""))
                    if not last_date:
                        last_date = str(doc.get("Date Collected", ""))

    filtered_snippets.append((
        " | ".join(matched_sentences),
        " | ".join(urls),
        " | ".join(matched_pages),
        doc_type,
        pub_date,
        last_date
    ))

# Append results
filtered_cols = [
    "Filtered Content", "URL", "Page number",
    "Document Type", "Publication Date", "Last updated Date"
]

for col, values in zip(filtered_cols, zip(*filtered_snippets)):
    df_17.loc[:, col] = values

# Save to file
output_path = os.path.dirname(os.path.abspath(__file__))
df_17.to_csv(os.path.join(output_path, "sdg17_questions_with_filtered_content.csv"), index=False)
print("SDG 17 with filtered content saved.")
