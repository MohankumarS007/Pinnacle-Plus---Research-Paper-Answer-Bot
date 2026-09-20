"""
Dataset Builder and Validator for Pinnacle Plus Capstone.
Downloads 15 seminal Generative AI / LLM research papers from arXiv,
generates data/metadata.csv, and validates the entire dataset.
"""

import os
import sys
import time
import csv
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PAPERS_DIR = os.path.join(DATA_DIR, 'papers')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

# 15 Curated Seminal Research Papers covering all core GenAI/LLM topics
CURATED_PAPERS = [
    {
        'paper_id': '1706.03762',
        'title': 'Attention Is All You Need',
        'category': 'Transformer Architecture',
        'year': 2017,
        'authors': 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
        'source': 'https://arxiv.org/abs/1706.03762',
        'file_name': '1706.03762_Attention_Is_All_You_Need.pdf'
    },
    {
        'paper_id': '1810.04805',
        'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
        'category': 'Language Representation',
        'year': 2018,
        'authors': 'Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova',
        'source': 'https://arxiv.org/abs/1810.04805',
        'file_name': '1810.04805_BERT.pdf'
    },
    {
        'paper_id': '2005.14165',
        'title': 'Language Models are Few-Shot Learners',
        'category': 'Large Language Models',
        'year': 2020,
        'authors': 'Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, et al.',
        'source': 'https://arxiv.org/abs/2005.14165',
        'file_name': '2005.14165_GPT3_Few_Shot_Learners.pdf'
    },
    {
        'paper_id': '1908.10084',
        'title': 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
        'category': 'Embeddings and Semantic Search',
        'year': 2019,
        'authors': 'Nils Reimers, Iryna Gurevych',
        'source': 'https://arxiv.org/abs/1908.10084',
        'file_name': '1908.10084_Sentence_BERT.pdf'
    },
    {
        'paper_id': '2004.04906',
        'title': 'Dense Passage Retrieval for Open-Domain Question Answering',
        'category': 'Information Retrieval',
        'year': 2020,
        'authors': 'Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih',
        'source': 'https://arxiv.org/abs/2004.04906',
        'file_name': '2004.04906_Dense_Passage_Retrieval.pdf'
    },
    {
        'paper_id': '2005.11401',
        'title': 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
        'category': 'RAG Foundation',
        'year': 2020,
        'authors': 'Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, et al.',
        'source': 'https://arxiv.org/abs/2005.11401',
        'file_name': '2005.11401_Retrieval_Augmented_Generation.pdf'
    },
    {
        'paper_id': '2107.13586',
        'title': 'Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing',
        'category': 'Prompting Methods',
        'year': 2021,
        'authors': 'Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig',
        'source': 'https://arxiv.org/abs/2107.13586',
        'file_name': '2107.13586_Prompting_Survey.pdf'
    },
    {
        'paper_id': '2201.11903',
        'title': 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
        'category': 'Reasoning and CoT',
        'year': 2022,
        'authors': 'Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, Denny Zhou',
        'source': 'https://arxiv.org/abs/2201.11903',
        'file_name': '2201.11903_Chain_of_Thought_Reasoning.pdf'
    },
    {
        'paper_id': '2302.04761',
        'title': 'Toolformer: Language Models Can Teach Themselves to Use Tools',
        'category': 'Tool Use and Function Calling',
        'year': 2023,
        'authors': 'Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom',
        'source': 'https://arxiv.org/abs/2302.04761',
        'file_name': '2302.04761_Toolformer.pdf'
    },
    {
        'paper_id': '2210.03629',
        'title': 'ReAct: Synergizing Reasoning and Acting in Language Models',
        'category': 'AI Agents',
        'year': 2022,
        'authors': 'Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao',
        'source': 'https://arxiv.org/abs/2210.03629',
        'file_name': '2210.03629_ReAct_Reasoning_And_Acting.pdf'
    },
    {
        'paper_id': '2307.03172',
        'title': 'Lost in the Middle: How Language Models Use Long Contexts',
        'category': 'Long-Context Behavior',
        'year': 2023,
        'authors': 'Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang',
        'source': 'https://arxiv.org/abs/2307.03172',
        'file_name': '2307.03172_Lost_in_the_Middle.pdf'
    },
    {
        'paper_id': '2310.11511',
        'title': 'Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection',
        'category': 'Advanced RAG',
        'year': 2023,
        'authors': 'Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi',
        'source': 'https://arxiv.org/abs/2310.11511',
        'file_name': '2310.11511_Self_RAG.pdf'
    },
    {
        'paper_id': '2312.10997',
        'title': 'Retrieval-Augmented Generation for Large Language Models: A Survey',
        'category': 'Modular RAG Survey',
        'year': 2023,
        'authors': 'Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang',
        'source': 'https://arxiv.org/abs/2312.10997',
        'file_name': '2312.10997_RAG_Comprehensive_Survey.pdf'
    },
    {
        'paper_id': '2309.15217',
        'title': 'RAGAS: Automated Evaluation of Retrieval Augmented Generation',
        'category': 'LLM and RAG Evaluation',
        'year': 2023,
        'authors': 'Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert',
        'source': 'https://arxiv.org/abs/2309.15217',
        'file_name': '2309.15217_RAGAS_Automated_Evaluation.pdf'
    },
    {
        'paper_id': '2311.05232',
        'title': 'A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions',
        'category': 'Hallucination and Reliability',
        'year': 2023,
        'authors': 'Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, et al.',
        'source': 'https://arxiv.org/abs/2311.05232',
        'file_name': '2311.05232_Hallucination_in_LLMs_Survey.pdf'
    }
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def download_pdf(arxiv_id: str, dest_path: str, max_retries: int = 3) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        size = os.path.getsize(dest_path)
        print(f"  [Already exists] {os.path.basename(dest_path)} ({size:,} bytes)")
        return True

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Downloading {arxiv_id} (Attempt {attempt}/{max_retries})...")
            resp = requests.get(url, headers=HEADERS, timeout=45, stream=True)
            if resp.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                file_size = os.path.getsize(dest_path)
                if file_size > 10000:
                    print(f"  [Downloaded] {os.path.basename(dest_path)} ({file_size:,} bytes)")
                    time.sleep(2)  # Polite crawl delay
                    return True
                else:
                    print(f"  [Warning] Downloaded file too small ({file_size} bytes), retrying...")
            else:
                print(f"  [HTTP {resp.status_code}] Failed to download {url}")
        except Exception as e:
            print(f"  [Error] {e}")
        time.sleep(3)
    return False

def build_dataset():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    print(f"=== STEP 2: Downloading {len(CURATED_PAPERS)} Curated Research Papers to {PAPERS_DIR} ===")
    
    downloaded = 0
    for idx, paper in enumerate(CURATED_PAPERS, 1):
        dest = os.path.join(PAPERS_DIR, paper['file_name'])
        print(f"[{idx}/{len(CURATED_PAPERS)}] {paper['title']}")
        if download_pdf(paper['paper_id'], dest):
            downloaded += 1
        else:
            print(f"  FAILED to download {paper['paper_id']}")

    print(f"\nTotal PDFs successfully downloaded/verified: {downloaded}/{len(CURATED_PAPERS)}")

    print(f"\n=== STEP 3: Creating {METADATA_PATH} ===")
    fieldnames = ['paper_id', 'title', 'category', 'year', 'authors', 'source', 'file_name']
    with open(METADATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paper in CURATED_PAPERS:
            writer.writerow({k: paper[k] for k in fieldnames})
    print(f"metadata.csv successfully written with {len(CURATED_PAPERS)} rows.")

def validate_dataset() -> bool:
    print(f"\n=== STEP 4: DATASET VALIDATION ===")
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: metadata.csv does not exist at {METADATA_PATH}")
        return False

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    num_metadata_rows = len(reader)
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    num_pdfs = len(pdf_files)

    errors = []
    seen_ids = set()
    matched_files = 0

    for row in reader:
        pid = row.get('paper_id')
        fname = row.get('file_name')

        if pid in seen_ids:
            errors.append(f"Duplicate paper_id: {pid}")
        seen_ids.add(pid)

        expected_file = os.path.join(PAPERS_DIR, fname)
        if not os.path.exists(expected_file):
            errors.append(f"Metadata references missing file: {fname}")
        elif os.path.getsize(expected_file) < 10000:
            errors.append(f"PDF file corrupted or empty (<10KB): {fname}")
        else:
            matched_files += 1

    print("=" * 45)
    print("        DATASET VALIDATION REPORT")
    print("=" * 45)
    print(f"Total selected papers:   {len(CURATED_PAPERS)}")
    print(f"Total PDFs in directory: {num_pdfs}")
    print(f"Metadata rows in CSV:    {num_metadata_rows}")
    print(f"Matched valid PDFs:      {matched_files}")
    print(f"Missing PDFs:            {num_metadata_rows - matched_files}")
    print(f"Duplicate IDs:           {len(reader) - len(seen_ids)}")
    print("=" * 45)

    if errors or matched_files != 15 or num_metadata_rows != 15:
        print("VALIDATION STATUS: FAILED")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("VALIDATION STATUS: PASSED (All 15 papers verified)")
        return True

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        ok = validate_dataset()
        sys.exit(0 if ok else 1)
    else:
        build_dataset()
        ok = validate_dataset()
        sys.exit(0 if ok else 1)