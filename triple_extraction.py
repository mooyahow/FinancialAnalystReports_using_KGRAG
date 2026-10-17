
import os
import json
import time
import re
import pdfplumber
from llama_index.core import Document
import hashlib
import pandas as pd

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from neo4j import GraphDatabase, basic_auth

# Configuration
CONFIG = {
    "gemini_api_key": "-",  # 사용자 API 키 입력
    "gemini_model_name": "models/gemini-2.0-flash",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "data_folder_path": "-", # 사용자 데이터 경로로 변경
    "triples_output_directory": "-", # 트리플.txt 추출 경로 추가
    "neo4j_uri": "-", # 사용자 Neo4j URI 입력
    "neo4j_user": "neo4j",
    "chroma_persist_directory": "./chroma_db",
    "chroma_collection_name": "kg_triples_unified-mod",
    "api_call_delay_seconds": 2,
    "candidate_node_types": ["company", "industry", "metric", "broker", "analyst"], # 노드 타입 후보

    # 트리플 추출시에는 안써도 될 거 같습니다.
    "evaluation_input_excel_path": "-.xlsx", # 평가 질문 Excel 파일 경로
    "evaluation_output_excel_path": "-.xlsx", # 평가 결과 저장 Excel 파일 경로
    "excel_question_column_name": "-" # 질문이 있는 열의 이름
}

# Helper Function for ID Generation (ChromaDBManager 외부 또는 상단에 위치)
def generate_hash_from_text(text_content):
    """Generates a 16-character hex hash from text content."""
    if not isinstance(text_content, str):
        text_content = str(text_content)
    hash_object = hashlib.sha256(text_content.encode('utf-8'))
    return hash_object.hexdigest()[:16]

# Initialize APIs and Models
def configure_gemini(api_key):
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("Gemini API 키가 설정되지 않았습니다. CONFIG를 확인해주세요.")
        raise ValueError("Gemini API key not configured.")
    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
        raise

def load_generative_model(model_name):
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Gemini model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load Gemini model '{model_name}': {e}")
        raise

def load_embedding_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        print(f"Embedding model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load embedding model '{model_name}': {e}")
        raise

# Data Loading
def load_text_documents(folder_path):
    loaded_documents = []
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Folder not found or is not a directory: {folder_path}")
        return []
    print(f"'{folder_path}' 폴더에서 PDF 문서 로딩 시작...")
    pdf_file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_file_names:
        print(f"  '{folder_path}'에서 PDF 파일을 찾지 못했습니다.")
        return []
    print(f"  Processing {len(pdf_file_names)} PDF file(s)...")
    for file_name in pdf_file_names:
        file_path = os.path.join(folder_path, file_name)
        try:
            with pdfplumber.open(file_path) as pdf:
                print(f"    '{file_name}': 총 {len(pdf.pages)} 페이지. 처리 중...")
                for page_obj in pdf.pages:
                    page_text = page_obj.extract_text()
                    if page_text and page_text.strip():
                        doc = Document(
                            text=page_text,
                            metadata={
                                "file_name": file_name,
                                "page_number": page_obj.page_number,
                                "source_type": "pdf_page"
                            }
                        )
                        loaded_documents.append(doc)
        except Exception as e:
            print(f"PDF 파일 '{file_name}' 처리 중 오류: {e}")
    if loaded_documents:
        print(f"총 {len(loaded_documents)}개의 PDF 페이지가 로딩되었습니다.")
    else:
        print(f"'{folder_path}'에서 PDF 파일로부터 로딩할 수 있는 페이지를 찾지 못했습니다.")
    return loaded_documents

# Triple Extraction
class TripleExtractor:
    def __init__(self, gemini_model, candidate_node_types):
        self.model = gemini_model
        self.candidate_node_types = candidate_node_types

    def _build_triple_prompt(self, text_block):
        candidate_types_str = ", ".join(self.candidate_node_types)
        return f"""
        너는 증권사 리서치 리포트를 분석하는 금융 데이터 전문가이다.
        아래 텍스트는 애널리스트의 분석이 담긴 증권 리서치 리포트이다.
        너의 임무는 다음과 같다:
        1. 문장에서 의미 있는 "삼항(triple)" 정보를 추출하라.
           - triple 구조는 (subject: 개체, subject_node_type: subject의 노드 유형, subject_industry: subject가 company일 경우 산업명, relation: 지표나 행위 또는 의미, object: 관련된 값이나 개체, object_node_type: object의 노드 유형, object_industry: object가 company일 경우 산업명, type: 관계의 사실/의견 구분)이다.
        2. `subject_node_type`과 `object_node_type`은 다음 후보군 중에서 가장 적절한 것을 선택하거나, 없으면 null 또는 빈 문자열로 설정하라: [{candidate_types_str}]
        3. `subject` 또는 `object`의 `node_type`이 "company"로 식별될 경우, 텍스트 내에서 해당 회사의 산업(industry) 정보가 언급되었다면 `subject_industry` 또는 `object_industry` 필드에 해당 산업명을 추출하여 포함시켜라. 정보가 없다면 null 또는 빈 문자열로 설정하라.
        4. 각 triple의 `type`이 **객관적 사실(fact)**인지, 또는 **애널리스트의 주관적 의견(opinion)**인지 구분하라. 이 `type`은 관계(relation)의 속성이다.
        5. 결과는 JSON 리스트 형식으로 출력하라:
        ```json
        [
          {{
            "subject": "...", "subject_node_type": "company" or null, "subject_industry": "반도체체" or null,
            "relation": "...",
            "object": "...", "object_node_type": "industry" or null, "object_industry": null,
            "type": "fact" or "opinion"
          }}
        ]
        ```
        예시 입력 1:
        문장: "전자 산업의 대표 기업인 크래프톤은 2025년 1분기 매출이 1.2조원으로 전년 대비 10% 증가하였다. 이는 게임 산업의 성장세에 따른 것이다."

        → Triple 1: {{ "subject": "크래프톤", "subject_node_type": "company", "subject_industry": "게임", "relation": "2025년 1분기 매출", "object": "1.2조원", "object_node_type": "metric", "object_industry": null, "type": "fact" }}
        → Triple 2: {{ "subject": "2025년 1분기 매출", "subject_node_type": "metric", "subject_industry": null, "relation": "전년 대비 증가율", "object": "10%", "object_node_type": "metric", "object_industry": null, "type": "fact" }}
        → Triple 3: {{ "subject": "크래프톤", "subject_node_type": "company", "subject_industry": "게임", "relation": "성장 원인", "object": "게임 산업 성장세", "object_node_type": "industry", "object_industry": null, "type": "opinion" }}
        아래 본문을 분석하라:
        {text_block}
    """

    def _parse_json_response(self, text_response):
        try:
            match = re.search(r"```json\n(.*?)```", text_response, re.DOTALL)
            json_str = match.group(1) if match else text_response
            json_str = json_str.replace("'", '"').strip()
            json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            if not json_str.startswith("[") and json_str.startswith("{"):
                json_str = f"[{json_str}]"
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            parsed_obj = json.loads(json_str)
            return [parsed_obj] if isinstance(parsed_obj, dict) else parsed_obj
        except json.JSONDecodeError as e:
            print(f"JSON 정제 및 파싱 실패: {e}. 응답 일부: '{text_response[:200]}...'")
            return []
        except Exception as e:
            print(f"JSON 정제 중 예기치 않은 오류: {e}")
            return []

    def extract_triples_from_text(self, text_block):
        prompt = self._build_triple_prompt(text_block)
        try:
            response = self.model.generate_content(prompt)
            content = response.text.strip()
            triples = self._parse_json_response(content)
            return triples, content
        except Exception as e:
            print(f"Gemini API 호출 또는 응답 처리 실패: {e}")
            return [], ""

    def process_documents(self, documents, api_delay_sec, triples_output_dir=None): # triples_output_dir 파라미터 추가
        all_extracted_data = []
        total_docs = len(documents)

        if triples_output_dir:
            try:
                os.makedirs(triples_output_dir, exist_ok=True)
                print(f"추출된 트리플 저장 경로: '{os.path.abspath(triples_output_dir)}'")
            except Exception as e:
                print(f"트리플 출력 디렉토리 '{triples_output_dir}' 생성 실패: {e}")
                triples_output_dir = None # 생성 실패 시 저장 기능 비활성화

        for i, doc in enumerate(documents):
            file_name = doc.metadata.get("file_name", f"doc_unknown_{i}")
            page_num = doc.metadata.get("page_number") # PDF 페이지의 경우 page_number가 있음
            source_type = doc.metadata.get("source_type")
            source_info = f"{file_name}" + (f" (Page {page_num})" if page_num is not None else "")

            print(f"문서 {i+1}/{total_docs} ({source_info}) 처리 중...")
            try:
                text_to_process = doc.text
                if not text_to_process or not text_to_process.strip():
                    print(f"  문서 ({source_info}) 내용이 비어있어 건너뜁니다.")
                    triples, raw_response = [], "Empty or whitespace content"
                else:
                    triples, raw_response = self.extract_triples_from_text(text_to_process)

                processed_triples = []
                if isinstance(triples, list):
                    for t_item in triples:
                        if isinstance(t_item, dict):
                            processed_triples.append({
                                "subject": t_item.get("subject"),
                                "subject_node_type": t_item.get("subject_node_type"),
                                "subject_industry": t_item.get("subject_industry"),
                                "relation": t_item.get("relation"),
                                "object": t_item.get("object"),
                                "object_node_type": t_item.get("object_node_type"),
                                "object_industry": t_item.get("object_industry"),
                                "type": t_item.get("type")
                            })
                        else:
                            print(f"  문서 ({source_info})에서 추출된 triple 항목이 dict가 아닙니다: {t_item}")
                else:
                     print(f"  문서 ({source_info})에서 추출된 triples 결과가 list가 아닙니다: {triples}")

                if triples_output_dir and processed_triples: # 저장 경로가 있고, 추출된 트리플이 있을 경우
                    base_name, _ = os.path.splitext(file_name) # 확장자 제거
                    # 페이지 번호가 있는 경우 파일명에 포함 (PDF 페이지)
                    page_suffix = f"_p{page_num}" if page_num is not None else ""
                    output_filename_txt = f"{base_name}{page_suffix}_triples.txt"
                    output_filepath = os.path.join(triples_output_dir, output_filename_txt)

                    try:
                        with open(output_filepath, 'w', encoding='utf-8') as f_out:
                            json.dump(processed_triples, f_out, ensure_ascii=False, indent=4)
                        print(f"  추출된 Triple {len(processed_triples)}개를 '{output_filepath}'에 저장했습니다.")
                    except IOError as e_io:
                        print(f"  파일 저장 실패 '{output_filepath}': {e_io}")
                    except Exception as e_gen:
                        print(f"  Triple 저장 중 예기치 않은 오류 발생 '{output_filepath}': {e_gen}")
                # 파일 저장 로직 끝

                all_extracted_data.append({
                    "file_name": file_name,
                    "page_number": page_num,
                    "source_type": source_type,
                    "triples": processed_triples,
                    "raw_gemini_response": raw_response
                })
                print(f"  문서 ({source_info}): Triple {len(processed_triples)}개 추출 완료.")
                time.sleep(api_delay_sec)
            except Exception as e:
                print(f"문서 ({source_info}) 처리 중 예기치 않은 오류: {e}")
                all_extracted_data.append({
                    "file_name": file_name,
                    "page_number": page_num,
                    "source_type": source_type,
                    "triples": [],
                    "raw_gemini_response": f"Error processing document: {e}"
                })
        return all_extracted_data

# Neo4j Graph Database
class Neo4jManager:
    def __init__(self, uri, user, password):
        if not all([uri, user, password]) or \
           uri == "YOUR_NEO4J_URI" or password == "YOUR_NEO4J_PASSWORD":
            print("Neo4j 접속 정보가 설정되지 않았습니다. CONFIG를 확인해주세요.")
            self.driver = None
            raise ValueError("Neo4j connection info not configured.")
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self.driver.verify_connectivity()
            print("Neo4j 드라이버 초기화 및 연결 성공.")
        except Exception as e:
            print(f"Neo4j 드라이버 초기화 실패: {e}")
            self.driver = None
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j 드라이버 연결이 종료되었습니다.")

    def _insert_triple_tx(self, tx, subject, subject_node_type, subject_industry,
                          relation_name, obj, object_node_type, object_industry,
                          relation_fact_opinion_type):
        query = """
        MERGE (s:Entity {name: $subject})
        FOREACH (ignore IN CASE WHEN $subject_node_type IS NOT NULL AND trim($subject_node_type) <> '' THEN [1] ELSE [] END |
            SET s.node_type = $subject_node_type
        )
        FOREACH (ignore IN CASE WHEN $subject_node_type = 'company' AND $subject_industry IS NOT NULL AND trim($subject_industry) <> '' THEN [1] ELSE [] END |
            SET s.industry = $subject_industry
        )
        MERGE (o:Entity {name: $object})
        FOREACH (ignore IN CASE WHEN $object_node_type IS NOT NULL AND trim($object_node_type) <> '' THEN [1] ELSE [] END |
            SET o.node_type = $object_node_type
        )
        FOREACH (ignore IN CASE WHEN $object_node_type = 'company' AND $object_industry IS NOT NULL AND trim($object_industry) <> '' THEN [1] ELSE [] END |
            SET o.industry = $object_industry
        )
        MERGE (s)-[rel:RELATION {type: $relation_fact_opinion_type}]->(o)
        SET rel.name = $relation_name
        """
        params = {
            "subject": str(subject) if subject is not None else "Unknown Subject",
            "subject_node_type": str(subject_node_type) if subject_node_type else None,
            "subject_industry": str(subject_industry) if subject_industry else None,
            "relation_name": str(relation_name) if relation_name is not None else "Unknown Relation",
            "object": str(obj) if obj is not None else "Unknown Object",
            "object_node_type": str(object_node_type) if object_node_type else None,
            "object_industry": str(object_industry) if object_industry else None,
            "relation_fact_opinion_type": str(relation_fact_opinion_type) if relation_fact_opinion_type else "unknown"
        }
        tx.run(query, **params)

    def store_triples_in_neo4j(self, all_extracted_data):
        if not self.driver:
            print("Neo4j 드라이버가 초기화되지 않아 저장을 건너뜁니다.")
            return 0
        total_inserted = 0
        with self.driver.session() as session:
            for doc_data in all_extracted_data:
                file_name = doc_data.get("file_name", "unknown_file")
                page_num = doc_data.get("page_number")
                source_id = f"{file_name}" + (f" (Page {page_num})" if page_num is not None else "")
                triples_to_insert = doc_data.get("triples", [])
                if not isinstance(triples_to_insert, list): continue
                for t_item in triples_to_insert:
                    if not isinstance(t_item, dict): continue
                    try:
                        subject, s_node_type, s_industry = t_item.get("subject"), t_item.get("subject_node_type"), t_item.get("subject_industry")
                        relation_name, obj, o_node_type, o_industry = t_item.get("relation"), t_item.get("object"), t_item.get("object_node_type"), t_item.get("object_industry")
                        relation_fact_opinion_type = t_item.get("type")
                        if not all(val is not None for val in [subject, relation_name, obj, relation_fact_opinion_type]):
                            print(f"필수 필드 누락 triple 건너뜁니다: {t_item} (출처: {source_id})")
                            continue
                        session.execute_write(
                            self._insert_triple_tx, subject, s_node_type, s_industry,
                            relation_name, obj, o_node_type, o_industry, relation_fact_opinion_type
                        )
                        total_inserted += 1
                    except Exception as e:
                        print(f"Neo4j 트랜잭션 실패 (triple: {t_item}, 출처: {source_id}): {e}")
        print(f"총 {total_inserted}개의 triple에 대해 Neo4j 저장 시도 완료.")
        return total_inserted

    # Condition i) subject가 "company"일 때 사용: s_industry가 같은 다른 s_new를 찾고, 그 s_new의 o_new 중 o_target_node_type과 일치하는 것을 찾음
    def expand_kg_by_s_industry_and_o_type(self, s_name_to_exclude, s_industry_to_match, o_target_node_type, limit=10):
        if not self.driver:
            print("Neo4j 드라이버가 초기화되지 않아 KG 확장을 건너뜁니다.")
            return []
        if not all([s_industry_to_match, o_target_node_type]):
            print(f"KG 확장 (Company): industry ('{s_industry_to_match}') 또는 target object node_type ('{o_target_node_type}') 정보가 부족하여 확장을 건너뜁니다.")
            return []

        with self.driver.session() as session:
            query = """
            MATCH (s_new:Entity {industry: $s_industry_to_match})
            WHERE s_new.name <> $s_name_to_exclude
            MATCH (s_new)-[r_new:RELATION]->(o_new:Entity {node_type: $o_target_node_type})
            RETURN DISTINCT
                s_new.name AS subject,
                s_new.node_type AS subject_node_type,
                s_new.industry AS subject_industry,
                r_new.name AS relation,
                r_new.type AS type,
                o_new.name AS object,
                o_new.node_type AS object_node_type
            LIMIT $limit
            """
            result = session.run(query,
                                 s_name_to_exclude=s_name_to_exclude,
                                 s_industry_to_match=s_industry_to_match,
                                 o_target_node_type=o_target_node_type,
                                 limit=limit)
            return [record.data() for record in result]

    # NEW_METHOD for Condition ii) subject가 "company"가 아닐 때 사용: 원본 s에 직접 연결된 o_new 중 o_target_node_type과 일치하는 것을 찾음
    def expand_kg_direct_from_s_by_o_type(self, s_name, o_target_node_type, limit=10):
        if not self.driver:
            print("Neo4j 드라이버가 초기화되지 않아 KG 확장을 건너뜁니다.")
            return []
        if not all([s_name, o_target_node_type]):
            print(f"KG 확장 (Non-Company): subject name ('{s_name}') 또는 target object node_type ('{o_target_node_type}') 정보가 부족하여 확장을 건너뜁니다.")
            return []

        with self.driver.session() as session:
            query = """
            MATCH (s:Entity {name: $s_name})-[r:RELATION]->(o:Entity {node_type: $o_target_node_type})
            RETURN s.name AS subject, s.node_type AS subject_node_type, s.industry AS subject_industry,
                   r.name AS relation, r.type AS type,
                   o.name AS object, o.node_type AS object_node_type
            LIMIT $limit
            """
            result = session.run(query,
                                 s_name=s_name,
                                 o_target_node_type=o_target_node_type,
                                 limit=limit)
            return [record.data() for record in result]

# ChromaDB Vector Store (ChromaDBManager 클래스)
class ChromaDBManager:
    def __init__(self, embedding_model, persist_directory, collection_name):
        self.embedding_model = embedding_model
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            print(f"ChromaDB 클라이언트 초기화 및 컬렉션 '{collection_name}' 준비 완료.")
        except Exception as e:
            print(f"ChromaDB 초기화 실패: {e}")
            self.client = None; self.collection = None; raise

    def prepare_data_for_chroma(self, all_extracted_data):
        texts, metadatas = [], []
        for doc_data in all_extracted_data:
            file_name = doc_data.get("file_name", "unknown_file")
            page_number = doc_data.get("page_number")
            source_type_from_doc = doc_data.get("source_type")
            triples_in_doc = doc_data.get("triples", [])
            if not isinstance(triples_in_doc, list): continue
            for t_item in triples_in_doc:
                if not isinstance(t_item, dict): continue
                subject,relation,obj = t_item.get("subject"),t_item.get("relation"),t_item.get("object")
                if not all(val is not None for val in [subject, relation, obj]): continue
                sentence = f"{subject}의 {relation}은(는) {obj}이다."
                texts.append(sentence)
                meta_entry = {
                    "subject": subject, "relation": relation, "object": obj,
                    "type": t_item.get("type") or "unknown",
                    "source_file": file_name,
                    "subject_node_type": t_item.get("subject_node_type") or "N/A",
                    "subject_industry": t_item.get("subject_industry") or "N/A",
                    "object_node_type": t_item.get("object_node_type") or "N/A",
                    "source_type": source_type_from_doc or "unknown_source"
                }
                if page_number is not None: meta_entry["source_page"] = page_number
                metadatas.append(meta_entry)
        if texts:
            print(f"총 {len(texts)}개의 triple 문장을 ChromaDB용으로 변환했습니다 (중복 제거 전).")
            if metadatas: print(f"  첫 번째 triple의 메타데이터 예시 (중복 제거 전): {metadatas[0]}")
        else: print(f"ChromaDB용으로 변환할 triple 문장이 없습니다.")
        return texts, metadatas

    def store_in_chroma(self, texts, metadatas):
        if not self.collection: print("ChromaDB 컬렉션이 초기화되지 않아 저장을 건너뜁니다."); return 0
        if not texts: print("ChromaDB에 저장할 데이터가 없습니다 (입력 texts 리스트 비어있음)."); return 0
        unique_items = {}
        ordered_unique_texts = []
        for i, txt in enumerate(texts):
            if txt not in unique_items: unique_items[txt] = metadatas[i]; ordered_unique_texts.append(txt)
        if not ordered_unique_texts: print("고유한 텍스트가 없어 ChromaDB에 추가할 내용이 없습니다."); return 0
        final_texts = ordered_unique_texts
        final_metadatas = [unique_items[txt] for txt in final_texts]
        print(f"원본 {len(texts)} 문장 중 {len(final_texts)} 고유 문장으로 ChromaDB 저장 준비.")
        try:
            embeddings = self.embedding_model.encode(final_texts).tolist()
            ids = [f"triple_{generate_hash_from_text(txt)}" for txt in final_texts]
            if not (len(final_texts) == len(final_metadatas) == len(embeddings) == len(ids)):
                print(f"최종 데이터 리스트 길이 불일치. 저장 중단."); return 0
            existing_ids = set(self.collection.get(ids=ids, include=[])['ids'])
            new_texts_add, new_embed_add, new_meta_add, new_ids_add = [], [], [], []
            for i, current_id in enumerate(ids):
                if current_id not in existing_ids:
                    new_texts_add.append(final_texts[i]); new_embed_add.append(embeddings[i]);
                    new_meta_add.append(final_metadatas[i]); new_ids_add.append(current_id)
            if not new_ids_add: print("ChromaDB에 추가할 새로운 triple이 없습니다 (모두 이미 존재)."); return 0
            self.collection.add(documents=new_texts_add, embeddings=new_embed_add, metadatas=new_meta_add, ids=new_ids_add)
            print(f"ChromaDB에 {len(new_ids_add)}개의 신규 triple 문장이 저장되었습니다.")
            return len(new_ids_add)
        except Exception as e:
            print(f"ChromaDB 저장 실패: {e}")
            if 'final_metadatas' in locals() and final_metadatas: print(f"  오류 발생시 첫 메타데이터: {final_metadatas[0] if final_metadatas else 'N/A'}")
            return 0

    def query_similar_triples(self, user_question, n_results=5):
        if not self.collection: return {"documents": [[]], "metadatas": [[]]}
        try:
            q_embed = self.embedding_model.encode([user_question]).tolist()
            return self.collection.query(query_embeddings=q_embed, n_results=n_results, include=["documents", "metadatas"])
        except Exception as e: print(f"ChromaDB 검색 실패: {e}"); return {"documents": [[]], "metadatas": [[]]}

# RAG - Retrieval Augmented Generation
class RAGSystem:
    def __init__(self, gemini_model, chroma_manager, neo4j_manager):
        self.gemini_model = gemini_model
        self.chroma_manager = chroma_manager
        self.neo4j_manager = neo4j_manager

    def _build_answer_prompt(self, user_question, chroma_docs, kg_triples):
        return f"""너는 금융 리서치 전문가이다.
        사용자의 질문: "{user_question}"
        다음은 관련된 보고서에서 추출된 triple 정보이다:
        - ChromaDB 유사 문장들 (Vector Search 기반):
        {chr(10).join(["• " + doc for doc in chroma_docs]) if chroma_docs else "관련된 유사 문장이 없습니다."}

        - Neo4j 지식그래프 기반 확장 정보:
        {chr(10).join([
            f"• {t.get('subject', 'N/A')} (Type: {t.get('subject_node_type', 'N/A')}, Industry: {t.get('subject_industry', 'N/A')}) "
            f"- [{t.get('relation', 'N/A')} (Fact/Opinion: {t.get('type', 'N/A')})] -> "
            f"{t.get('object', 'N/A')} (Type: {t.get('object_node_type', 'N/A')})"
            for t in kg_triples
        ]) if kg_triples else "  • 관련된 지식그래프 정보가 없습니다."}

        위의 정보를 바탕으로, 사용자 질문에 대해 신뢰할 수 있는 자연어 답변을 생성하라.
        가능하면 수치를 구체적으로 언급하고, 예측이나 의견은 분리해서 표현하라.
        만약 정보가 부족하여 답변하기 어렵다면, 정보가 부족하다고 솔직하게 말하라.
        """

    def get_answer(self, user_question, n_chroma_results=5, n_kg_expansion_limit=10):
        if not self.chroma_manager:
            print("RAGSystem: ChromaDBManager가 초기화되지 않았습니다.")
            return "정보 검색 시스템에 문제가 있어 답변을 생성할 수 없습니다 (ChromaDB 오류)."

        chroma_search_results = self.chroma_manager.query_similar_triples(user_question, n_results=n_chroma_results)
        retrieved_docs = chroma_search_results.get("documents", [[]])[0]
        retrieved_metadatas = chroma_search_results.get("metadatas", [[]])[0]

        print("\n유사한 triple 문장 Top-K (ChromaDB):")
        if retrieved_docs:
            for i, r_doc in enumerate(retrieved_docs):
                print(f"• {r_doc}")
                # 메타데이터 함께 출력 (디버깅/확인용)
                if i < len(retrieved_metadatas) and retrieved_metadatas[i]:
                    meta_info = retrieved_metadatas[i]
                    print(f"  ㄴ Metadata: {meta_info.get('source_file')}, Page: {meta_info.get('source_page', 'N/A')}, ObjType: {meta_info.get('object_node_type')}")
        else:
            print("  • ChromaDB에서 유사 문장을 찾지 못했습니다.")

        expanded_kg_triples = []
        # 확장 기준 중복 처리를 위한 set (어떤 로직을 탔는지도 키에 포함 가능)
        processed_expansion_criteria = set()

        if self.neo4j_manager and self.neo4j_manager.driver:
            if not retrieved_metadatas:
                print("ChromaDB 검색 결과의 메타데이터가 없어 KG 확장을 건너뜁니다.")
            else:
                print(f"\nKG 확장 시도 (ChromaDB 메타데이터 기반 조건부 확장):")
                for chroma_meta in retrieved_metadatas:
                    if not isinstance(chroma_meta, dict): continue

                    s_name_from_chroma = chroma_meta.get("subject")
                    s_node_type_from_chroma = chroma_meta.get("subject_node_type")
                    s_industry_from_chroma = chroma_meta.get("subject_industry")
                    o_type_from_chroma = chroma_meta.get("object_node_type")

                    if not all([s_name_from_chroma, o_type_from_chroma]): # 최소한의 정보 확인
                        # print(f"  - KG 확장 건너뜀 (Subject 또는 Object Type 정보 부족): {chroma_meta}")
                        continue

                    current_triples_from_kg = []
                    expansion_logic_type = ""

                    if s_node_type_from_chroma == "company":
                        expansion_logic_type = "company_industry_expansion"
                        if s_industry_from_chroma and s_industry_from_chroma != "N/A":
                            expansion_key = (s_name_from_chroma, s_industry_from_chroma, o_type_from_chroma, expansion_logic_type)
                            if expansion_key not in processed_expansion_criteria:
                                processed_expansion_criteria.add(expansion_key)
                                print(f"   회사 로직 확장: s_name='{s_name_from_chroma}', s_industry='{s_industry_from_chroma}', o_type='{o_type_from_chroma}'")
                                current_triples_from_kg = self.neo4j_manager.expand_kg_by_s_industry_and_o_type(
                                    s_name_to_exclude=s_name_from_chroma,
                                    s_industry_to_match=s_industry_from_chroma,
                                    o_target_node_type=o_type_from_chroma,
                                    limit=n_kg_expansion_limit
                                )
                        else:
                            print(f"  - 회사 ('{s_name_from_chroma}')이지만 산업 정보 ('{s_industry_from_chroma}')가 유효하지 않아 해당 확장 건너뜁.")
                    else: # "company"가 아닌 경우
                        expansion_logic_type = "non_company_direct_expansion"
                        expansion_key = (s_name_from_chroma, o_type_from_chroma, expansion_logic_type)
                        if expansion_key not in processed_expansion_criteria:
                            processed_expansion_criteria.add(expansion_key)
                            print(f"  비회사 로직 확장: s_name='{s_name_from_chroma}', o_type='{o_type_from_chroma}'")
                            current_triples_from_kg = self.neo4j_manager.expand_kg_direct_from_s_by_o_type(
                                s_name=s_name_from_chroma,
                                o_target_node_type=o_type_from_chroma,
                                limit=n_kg_expansion_limit
                            )

                    if current_triples_from_kg:
                        expanded_kg_triples.extend(current_triples_from_kg)

        elif self.neo4j_manager is None:
             print("Neo4j Manager가 설정되지 않아 KG 확장을 건너뜁니다.")

        if expanded_kg_triples:
            seen_triples = set()
            unique_expanded_kg_triples = []
            for triple in expanded_kg_triples:
                try: identifier = frozenset(triple.items())
                except TypeError: identifier = str(triple) # Fallback for unhashable items
                if identifier not in seen_triples:
                    seen_triples.add(identifier); unique_expanded_kg_triples.append(triple)
            expanded_kg_triples = unique_expanded_kg_triples
            print(f"Neo4j에서 확장된 unique triple 수 (최종): {len(expanded_kg_triples)}")

        if expanded_kg_triples:
             print("--- 확장된 최종 KG Triples 예시 (최대 3개) ---")
             for t_kg in expanded_kg_triples[:3]:
                 print(f"  • {t_kg.get('subject')} (Type: {t_kg.get('subject_node_type')}, Ind: {t_kg.get('subject_industry')}) "
                       f"-[{t_kg.get('relation')}, Fact/Op: {t_kg.get('type')}]-> "
                       f"{t_kg.get('object')} (Type: {t_kg.get('object_node_type')})")
             if len(expanded_kg_triples) > 3: print(f"  ... 외 {len(expanded_kg_triples)-3}개 더 있음")
             print("-------------------------------------")
        elif not retrieved_metadatas and not expanded_kg_triples :
            print("ChromaDB 검색 결과 및 KG 확장 정보가 모두 없습니다.")

        final_prompt = self._build_answer_prompt(user_question, retrieved_docs, expanded_kg_triples)
        print("final_prompt",final_prompt)
        print("+========================================+")
        try:
            response = self.gemini_model.generate_content(final_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini 답변 생성 실패: {e}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."

# NEW FUNCTIONS for Batch Evaluation
def load_questions_from_excel(file_path, column_name="question"):
    """Loads questions from a specified column in an Excel file."""
    try:
        df = pd.read_excel(file_path)
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in Excel file '{file_path}'.")
            return []
        questions = df[column_name].dropna().astype(str).tolist()
        print(f"Loaded {len(questions)} questions from '{file_path}', column '{column_name}'.")
        return questions
    except FileNotFoundError:
        print(f"Error: Input Excel file not found at '{file_path}'.")
        return []
    except Exception as e:
        print(f"Error loading questions from Excel '{file_path}': {e}")
        return []

def save_results_to_excel(results_data, output_file_path):
    """Saves a list of dictionaries (e.g., [{'Question': q, 'Answer': a}]) to an Excel file."""
    if not results_data:
        print("No results to save to Excel.")
        return
    try:
        df = pd.DataFrame(results_data)
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string for current dir
            print(f"  Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        df.to_excel(output_file_path, index=False, engine='openpyxl') # Requires openpyxl
        print(f"Successfully saved {len(df)} results to '{output_file_path}'.")
    except Exception as e:
        print(f"Error saving results to Excel '{output_file_path}': {e}")

# Helper function to initialize models and APIs (as defined before)
def initialize_models_and_apis(config_params):
    print("\n--- 단계 1 (공통): API 및 모델 초기화 ---")
    try:
        if config_params["gemini_api_key"] == "YOUR_GEMINI_API_KEY" or not config_params["gemini_api_key"]:
            raise ValueError("Gemini API key not configured.")
        configure_gemini(config_params["gemini_api_key"])
        gemini_model = load_generative_model(config_params["gemini_model_name"])
        embedding_model = load_embedding_model(config_params["embedding_model_name"])
        print("공통 컴포넌트 초기화 완료.")
        return gemini_model, embedding_model
    except Exception as e: print(f"API/모델 초기화 중 심각 오류: {e}. 종료."); raise SystemExit()

# Pipeline 1: Data Processing and Storage (as defined before)
def run_data_processing_pipeline(config_params, gemini_model, embedding_model):
    print("\n>>>>>>>>>> 데이터 처리 및 저장 파이프라인 시작 >>>>>>>>>>")
    # (이전 답변의 run_data_processing_pipeline 전체 코드와 동일)
    print("\n--- 단계 2: 데이터 로딩 ---")
    documents = load_text_documents(config_params["data_folder_path"])
    if not documents: print("로딩된 문서가 없어 데이터 처리를 중단합니다."); return
    print("\n--- 단계 3: Triple 추출 ---")
    extractor = TripleExtractor(gemini_model, config_params["candidate_node_types"])
    all_triples_data = extractor.process_documents(documents, config_params["api_call_delay_seconds"], config_params.get("triples_output_directory"))
    if not all_triples_data: print("추출된 트리플 데이터가 없어 DB 저장을 건너뜁니다."); return

    neo4j_manager_proc = None
    try:
        neo4j_manager_proc = Neo4jManager(config_params["neo4j_uri"], config_params["neo4j_user"], config_params["neo4j_password"])
        if neo4j_manager_proc.driver: print("\n--- 단계 4: Neo4j에 Triple 저장 ---"); neo4j_manager_proc.store_triples_in_neo4j(all_triples_data)
        else: neo4j_manager_proc = None
    except ValueError as ve: print(f"\nNeo4j 설정 오류 (처리용): {ve}.")
    except Exception as e: print(f"\nNeo4j 처리 중 예기치 않은 오류: {e}")
    finally:
        if neo4j_manager_proc and neo4j_manager_proc.driver: neo4j_manager_proc.close()

    chroma_db_manager_proc = None
    try:
        chroma_db_manager_proc = ChromaDBManager(embedding_model, config_params["chroma_persist_directory"], config_params["chroma_collection_name"])
        if chroma_db_manager_proc.collection:
            print("\n--- 단계 5: ChromaDB에 Triple 임베딩 및 저장 ---")
            texts,metas = chroma_db_manager_proc.prepare_data_for_chroma(all_triples_data)
            chroma_db_manager_proc.store_in_chroma(texts,metas)
    except Exception as e: print(f"ChromaDB 처리 중 오류: {e}")
    print("\n<<<<<<<<<< 데이터 처리 및 저장 파이프라인 완료 <<<<<<<<<<")

# Pipeline 2: Querying with RAG (as defined before, but ensure it handles no DB managers)
def run_query_rag_system(config_params, gemini_model, embedding_model, user_question):
    print(f"\n>>>>>>>>>> RAG 질의응답 시스템 시작 (질문: {user_question}) >>>>>>>>>>")
    # (이전 답변의 run_query_rag_system 전체 코드와 동일 - 각 DB Manager 초기화 포함)
    neo4j_manager_rag, chroma_db_manager_rag = None, None
    try:
        neo4j_manager_rag = Neo4jManager(config_params["neo4j_uri"], config_params["neo4j_user"], config_params["neo4j_password"])
        if not neo4j_manager_rag.driver: neo4j_manager_rag = None
    except Exception as e: print(f"Neo4j Manager (RAG용) 초기화 오류: {e}"); neo4j_manager_rag = None
    try:
        chroma_db_manager_rag = ChromaDBManager(embedding_model, config_params["chroma_persist_directory"], config_params["chroma_collection_name"])
        if not chroma_db_manager_rag.collection: print("ChromaDB 컬렉션 (RAG용) 초기화 실패."); chroma_db_manager_rag = None # Ensure None
    except Exception as e: print(f"ChromaDB Manager (RAG용) 초기화 오류: {e}"); chroma_db_manager_rag = None

    if not chroma_db_manager_rag: # ChromaDB는 RAG에 필수
        print("ChromaDB가 준비되지 않아 RAG 질의응답을 진행할 수 없습니다.")
        if neo4j_manager_rag and neo4j_manager_rag.driver: neo4j_manager_rag.close()
        return "ChromaDB 오류로 답변 생성 불가"

    rag_system = RAGSystem(gemini_model, chroma_db_manager_rag, neo4j_manager_rag)
    final_answer = rag_system.get_answer(user_question)
    print("\n최종 생성 답변 (Gemini):"); print(final_answer)
    if neo4j_manager_rag and neo4j_manager_rag.driver: neo4j_manager_rag.close()
    print("\n<<<<<<<<<< RAG 질의응답 시스템 완료 <<<<<<<<<<")
    return final_answer

# NEW Pipeline 3: Batch Evaluation
def run_batch_evaluation_pipeline(config_params, gemini_model, embedding_model):
    """Loads questions from Excel, gets RAG answers, and saves them to a new Excel file."""
    print("\n>>>>>>>>>> 배치 평가 파이프라인 시작 >>>>>>>>>>")

    input_excel_path = config_params.get("evaluation_input_excel_path")
    output_excel_path = config_params.get("evaluation_output_excel_path")
    question_column = config_params.get("excel_question_column_name", "question") # 기본값 "question"

    if not input_excel_path or not output_excel_path:
        print("평가용 입력 또는 출력 Excel 파일 경로가 CONFIG에 설정되지 않았습니다. 평가를 중단합니다.")
        return

    questions = load_questions_from_excel(input_excel_path, column_name=question_column)
    if not questions:
        print("Excel 파일에서 질문을 불러오지 못했거나 질문이 없습니다. 평가를 종료합니다.")
        return

    # DB Managers for RAG system in batch mode - can be initialized once here
    batch_neo4j_manager = None
    try:
        batch_neo4j_manager = Neo4jManager(config_params["neo4j_uri"], config_params["neo4j_user"], config_params["neo4j_password"])
        if not batch_neo4j_manager.driver: batch_neo4j_manager = None
    except Exception as e: print(f"Neo4j Manager (배치 평가용) 초기화 오류: {e}"); batch_neo4j_manager = None

    batch_chroma_manager = None
    try:
        batch_chroma_manager = ChromaDBManager(
            embedding_model=embedding_model,
            persist_directory=config_params["chroma_persist_directory"],
            collection_name=config_params["chroma_collection_name"]
        )
        if not batch_chroma_manager.collection:
            print("ChromaDB 컬렉션 (배치 평가용) 초기화 실패. 평가를 중단합니다.")
            if batch_neo4j_manager and batch_neo4j_manager.driver: batch_neo4j_manager.close()
            return
    except Exception as e:
        print(f"ChromaDB Manager (배치 평가용) 초기화 오류: {e}")
        if batch_neo4j_manager and batch_neo4j_manager.driver: batch_neo4j_manager.close()
        return

    rag_system = RAGSystem(gemini_model, batch_chroma_manager, batch_neo4j_manager)

    evaluation_results = []
    total_questions = len(questions)
    for i, q_text in enumerate(questions):
        print(f"\n배치 질문 {i+1}/{total_questions} 처리 중: \"{q_text[:100]}{'...' if len(q_text)>100 else ''}\"")
        # rag_system.get_answer 내부에서 상세 로그가 출력됨
        answer_text = rag_system.get_answer(q_text)
        evaluation_results.append({'Question': q_text, 'Answer': answer_text})
        print(f"  질문 '{q_text[:50]}...'에 대한 답변 일부: {answer_text[:100]}...")
        # 배치 작업이므로 API 호출 딜레이는 RAGSystem 내부의 것을 따르거나, 여기서 추가 딜레이는 불필요할 수 있음

    save_results_to_excel(evaluation_results, output_excel_path)

    if batch_neo4j_manager and batch_neo4j_manager.driver:
        batch_neo4j_manager.close()
    print("\n<<<<<<<<<< 배치 평가 파이프라인 완료 <<<<<<<<<<")

# Main Execution Block
if __name__ == "__main__":
    print("--- 프로그램 시작 ---")
    # CONFIG 값 유효성 검사
    if CONFIG["gemini_api_key"] == "YOUR_GEMINI_API_KEY" or \
       CONFIG["neo4j_uri"] == "YOUR_NEO4J_URI" or \
       CONFIG["neo4j_password"] == "YOUR_NEO4J_PASSWORD":
        print("!!!!!!!!!!!!!!!!!!! 중요: CONFIG의 API 키 및 DB 정보를 실제 값으로 수정해주세요. !!!!!!!!!!!!!!!!!!!")
        # raise SystemExit("필수 CONFIG 값이 설정되지 않았습니다.") # 또는 필요시 프로그램 종료

    gemini_model, embedding_model = initialize_models_and_apis(CONFIG)

    # 실행 모드 선택
    # TODO: 아래 플래그를 필요에 따라 True/False로 설정하여 실행하세요.
    DO_DATA_PROCESSING = True   # 데이터 처리 및 저장 파이프라인 실행 여부
    DO_INTERACTIVE_QUERYING = False # 대화형 단일 질문 실행 여부
    DO_BATCH_EVALUATION = False   # 엑셀 기반 배치 평가 실행 여부

    if DO_DATA_PROCESSING:
        run_data_processing_pipeline(CONFIG, gemini_model, embedding_model)

    if DO_INTERACTIVE_QUERYING:
        print("\n--- 대화형 질의응답 모드 ---")
        if not DO_DATA_PROCESSING:
            print("데이터베이스가 이미 준비되어 있다고 가정합니다.")

        default_q_interactive = "삼성전자의 2025년 실적은 어떨까?"
        try:
            user_q_interactive = input(f"질문을 입력하세요 (기본값: '{default_q_interactive}'): ").strip()
            if not user_q_interactive: user_q_interactive = default_q_interactive
            run_query_rag_system(CONFIG, gemini_model, embedding_model, user_q_interactive)
        except EOFError:
            print("\n입력 스트림이 없어 기본 질문으로 질의응답을 실행합니다.")
            run_query_rag_system(CONFIG, gemini_model, embedding_model, default_q_interactive)

    if DO_BATCH_EVALUATION:
        print("\n--- 배치 평가 모드 ---")
        if not DO_DATA_PROCESSING and not DO_INTERACTIVE_QUERYING: # 만약 다른 모드가 전혀 실행 안됐다면
             print("배치 평가를 실행합니다. 데이터베이스가 이미 준비되어 있다고 가정.")
        run_batch_evaluation_pipeline(CONFIG, gemini_model, embedding_model)

    print("\n--- 모든 작업 완료 ---")
