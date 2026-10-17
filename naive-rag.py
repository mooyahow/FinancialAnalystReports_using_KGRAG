import os
import pdfplumber
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chromadb import PersistentClient
import google.generativeai as genai
import time
import hashlib
from sentence_transformers import SentenceTransformer

# === 1. 설정 ===
CONFIG = {
    "gemini_api_key": "-",  # 사용자 API 키 입력
    "gemini_model_name": "models/gemini-2.0-flash",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "data_folder_path": "", # PDF 파일이 있는 폴더 경로
    "chroma_persist_directory": "./chroma_db_naive_rag", # ChromaDB 저장 경로
    "chroma_collection_name": "naive-rag-collection", # ChromaDB 컬렉션 이름
    "api_call_delay_seconds": 2,

    "evaluation_input_excel_path": "", # 평가 질문 Excel 파일 경로
    "evaluation_output_excel_path": ".", # 평가 결과 저장 Excel 파일 경로
    "excel_question_column_name": "question" # 질문이 있는 열의 이름
}

# 청크 크기 (CONFIG에 포함할 수도 있습니다)
CHUNK_SIZE = 512

# === 2. Gemini 및 임베딩 모델 설정 ===

# Gemini API 키 설정
if CONFIG["gemini_api_key"] == "YOUR_GEMINI_API_KEY":
    print("경고: CONFIG['gemini_api_key']를 실제 Gemini API 키로 설정해주세요.")
genai.configure(api_key=CONFIG["gemini_api_key"])

# Gemini 생성 모델 초기화
llm_model = genai.GenerativeModel(CONFIG["gemini_model_name"])

# 임베딩 모델 및 토크나이저 초기화
embed_model = SentenceTransformer(CONFIG["embedding_model_name"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["embedding_model_name"]) # 정확히는 SentenceTransformer 내부에서 토크나이징 처리

# === 3. PDF 파싱 + 청크 분할 ===
def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        # 토큰들을 다시 문자열로 변환합니다.
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        # 빈 청크나 매우 짧은 청크는 제외할 수 있습니다.
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks

all_chunks = []
metadatas = []

# PDF 파일 경로 확인
pdf_folder_path = CONFIG["data_folder_path"]
if not os.path.isdir(pdf_folder_path):
    print(f"오류: PDF 폴더를 찾을 수 없습니다: {pdf_folder_path}")
    # exit()

print(f"PDF 파일 로딩 중... 경로: {pdf_folder_path}")
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder_path, filename)
        print(f"{filename} 파싱 중...")
        try:
            with pdfplumber.open(path) as pdf:
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    extracted_text = page.extract_text()
                    if extracted_text:
                        full_text += extracted_text + "\n"
                    else:
                        print(f"   {filename}의 {page_num + 1}페이지에서 텍스트를 추출할 수 없습니다.")

                if full_text.strip():
                    chunks = chunk_text(full_text, chunk_size=CHUNK_SIZE)
                    all_chunks.extend(chunks)
                    metadatas.extend([{"source": filename, "chunk_index": i} for i in range(len(chunks))])
                else:
                    print(f"   {filename}에서 추출된 텍스트가 없습니다.")
        except Exception as e:
            print(f"   {filename} 파싱 중 오류 발생: {e}")

if not all_chunks:
    print("오류: PDF에서 추출된 청크가 없습니다. PDF 파일 내용과 경로를 확인해주세요.")
    # exit()

print(f"총 {len(all_chunks)}개의 청크 생성 완료.")

# === 4. ChromaDB에 저장 ===
chroma_persist_dir = CONFIG["chroma_persist_directory"]
if not os.path.exists(chroma_persist_dir):
    os.makedirs(chroma_persist_dir)

client = PersistentClient(path=chroma_persist_dir)
collection_name = CONFIG["chroma_collection_name"]

try:
    collection = client.get_collection(name=collection_name)
    print(f"기존 ChromaDB 컬렉션 '{collection_name}'을 로드했습니다. 현재 {collection.count()}개의 문서가 있습니다.")

except Exception:
    print(f"ChromaDB 컬렉션 '{collection_name}'을 새로 생성합니다.")
    collection = client.create_collection(name=collection_name)

if all_chunks:
    print(f"ChromaDB에 청크 저장/업데이트 중 ({len(all_chunks)}개 대상)...")
    embeddings = embed_model.encode(all_chunks, convert_to_tensor=False, show_progress_bar=True).tolist()

    ids = []
    for i in range(len(all_chunks)):
        chunk_content = all_chunks[i]
        source_file = metadatas[i]['source']
        content_to_hash = f"content:{chunk_content}_source:{source_file}" # 요청대로 내용과 소스만 사용

        hashed_id = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        ids.append(hashed_id)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"저장/업데이트 완료: 현재 '{collection_name}' 컬렉션에 {collection.count()}개 청크가 저장되었습니다.")
else:
    print("저장할 청크가 없어 ChromaDB에 데이터를 추가하지 않았습니다.")

# === 5. 질문 로드 및 질의 ===
try:
    df = pd.read_excel(CONFIG["evaluation_input_excel_path"])
except FileNotFoundError:
    print(f"오류: 질문 파일을 찾을 수 없습니다: {CONFIG['evaluation_input_excel_path']}")
    df = pd.DataFrame()

question_column = CONFIG["excel_question_column_name"]
if df.empty or question_column not in df.columns:
    print(f"오류: Excel 파일을 로드할 수 없거나 '{question_column}' 열이 없습니다. CONFIG를 확인해주세요.")
    questions = []
else:
    questions = df[question_column].dropna().tolist()

answers = []
retrieved_contexts_for_eval = []

if not questions:
    print("처리할 질문이 없습니다.")
else:
    print(f"\n총 {len(questions)}개의 질문에 대해 답변 생성 시작...")
    for q_idx, q in enumerate(tqdm(questions, desc="Gemini 질의 중")):
        print(f"\n질문 {q_idx + 1}/{len(questions)}: {q}")

        q_embedding = embed_model.encode([q]).tolist()
        try:
            # 컬렉션에 문서가 있는지 확인
            if collection.count() == 0:
                print("   컨텍스트 검색 건너뛰기: ChromaDB 컬렉션이 비어 있습니다.")
                context_docs = []
                context_metadatas_list = []
            else:
                results = collection.query(query_embeddings=q_embedding, n_results=min(5, collection.count())) # 상위 5개 또는 컬렉션 전체 개수 중 작은 값
                context_docs = results["documents"][0] if results["documents"] else []
                context_metadatas_list = results["metadatas"][0] if results["metadatas"] else []

            if not context_docs:
                print("   검색된 컨텍스트가 없습니다.")
                context = "검색된 컨텍스트가 없습니다."
                retrieved_contexts_for_eval.append([{"doc": "없음", "meta": {}}])
            else:
                context = "\n\n---\n\n".join(context_docs) # 각 문서 사이에 구분선 추가
                retrieved_contexts_for_eval.append([{"doc": doc, "meta": meta} for doc, meta in zip(context_docs, context_metadatas_list)])
                print(f"   검색된 컨텍스트 소스: {[meta.get('source', 'N/A') for meta in context_metadatas_list]}")

        except Exception as e:
            print(f"   컨텍스트 검색 중 오류: {e}")
            context = "컨텍스트를 검색하지 못했습니다."
            retrieved_contexts_for_eval.append([{"doc": "오류", "meta": {"error": str(e)}}])

        prompt = f"""당신은 주어진 문서를 바탕으로 질문에 답변하는 금융 리서치 전문가이다.
위의 정보를 바탕으로, 사용자 질문에 대해 신뢰할 수 있는 자연어 답변을 생성하라.
가능하면 수치를 구체적으로 언급하고, 예측이나 의견은 분리해서 표현하라.
만약 정보가 부족하여 답변하기 어렵다면, 정보가 부족하다고 솔직하게 말하라.

[문서]
{context}

[질문]
{q}

[답변]"""

        try:
            if not CONFIG["gemini_api_key"] or CONFIG["gemini_api_key"] == "YOUR_GEMINI_API_KEY":
                answer = "Gemini API 키가 설정되지 않아 답변을 생성할 수 없습니다."
                print(f"   {answer}")
            else:
                response = llm_model.generate_content(prompt)
                answer = response.text.strip()
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    print(f"   Gemini API 응답 차단됨: {response.prompt_feedback.block_reason}")
                    answer = f"답변 생성 중 문제가 발생했습니다 (차단됨: {response.prompt_feedback.block_reason})."
        except Exception as e:
            print(f"   Gemini API 호출 중 오류 발생: {e}")
            answer = "Gemini API 호출 중 오류가 발생했습니다."

        answers.append(answer)
        # print(f"   답변: {answer[:100]}...") # 답변 일부만 출력 (필요시 주석 해제)
        print(f"   답변 생성됨.")

        time.sleep(CONFIG["api_call_delay_seconds"])

# === 6. 결과 저장 ===
if not questions:
    print("\n생성된 답변이 없어 결과를 저장하지 않습니다.")
else:
    df["generated_answer_naive_rag"] = answers

    output_path = CONFIG["evaluation_output_excel_path"]
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n결과 저장 완료: {output_path}")
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
        try:
            csv_output_path = output_path.replace(".xlsx", ".csv")
            df.to_csv(csv_output_path, index=False)
            print(f"Excel 저장 실패. CSV로 대신 저장했습니다: {csv_output_path}")
        except Exception as e_csv:
            print(f"CSV 저장 중에도 오류 발생: {e_csv}")

print("\n스크립트 실행 완료.")
