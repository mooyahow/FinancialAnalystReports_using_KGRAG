import pandas as pd
import re

# --- 설정 ---
EXCEL_FILE_PATH = "./---.xlsx"
QUESTION_COLUMN_NAME_FOR_LOGGING = 'question'
EXPECTED_ANSWER_VALUE_COLUMN_NAME = 'ground truth'
GENERATED_ANSWER_COLUMN_NAME = 'RAG@5'


# --- 도우미 함수 (공통 및 폴백용) ---
def _normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 단순 문자열 비교용 함수 (이전과 동일) ---
def extract_key_number_from_gt_simple(gt_text_raw):
    if not isinstance(gt_text_raw, str):
        gt_text_raw = str(gt_text_raw)
    match = re.search(r"([-]?[\d,.-]+)", gt_text_raw) # 음수 포함하도록 수정됨
    if match:
        return match.group(1).replace(",", "")
    return None

def normalize_gt_number_trailing_zeros_simple(num_str):
    if num_str is None: return None
    if '.' in num_str:
        while num_str.endswith('0'): num_str = num_str[:-1]
        if num_str.endswith('.'): num_str = num_str[:-1]
    return num_str

# --- 폴백: 단위 고려 비교용 함수들 (정규식에 음수 처리 강화) ---
def _parse_monetary_to_eokwon_fallback(value_str, unit_str):
    try:
        num = float(value_str) # float()는 "-7.4" 같은 음수 문자열 잘 처리함
        if "조" in unit_str: return num * 10000
        elif "억" in unit_str: return num
        elif "원" in unit_str and "억" not in unit_str and "조" not in unit_str: return num / 100000000
        return None
    except ValueError: return None

def parse_typed_value_for_fallback_gt(expected_value_text): # GT 값 파싱
    text = _normalize_text(expected_value_text)
    # 퍼센트 (음수 포함)
    m_percent = re.fullmatch(r"([-]?[\d.]+)\s*%", text)
    if m_percent:
        try: return (float(m_percent.group(1)), "percent")
        except ValueError: pass
    # 금액 (음수 포함)
    m_monetary = re.fullmatch(r"([-]?[\d.]+)\s*(조원|조|억원|억\s*원|억|원)", text)
    if m_monetary:
        value_str, unit_str_match = m_monetary.group(1), m_monetary.group(2)
        canonical_unit = "조" if "조" in unit_str_match else \
                         "억원" if "억원" in unit_str_match or ("억" in unit_str_match and "원" in unit_str_match) else \
                         "억" if "억" in unit_str_match else \
                         "원" if "원" in unit_str_match else None
        if canonical_unit:
            val_eokwon = _parse_monetary_to_eokwon_fallback(value_str, canonical_unit)
            if val_eokwon is not None: return (val_eokwon, "monetary_eokwon")
    # 배수 (음수 배수는 일반적이지 않지만, 형식상 허용)
    m_multiplier = re.fullmatch(r"([-]?[\d.]+)\s*배", text)
    if m_multiplier:
        try: return (float(m_multiplier.group(1)), "multiplier")
        except ValueError: pass
    # 백만 단위 수량 (음수 포함)
    m_million_unit = re.fullmatch(r"([-]?[\d.]+)\s*백만\s*(\w+)?", text)
    if m_million_unit:
        try:
            val = float(m_million_unit.group(1)) * 1000000
            unit_suffix = m_million_unit.group(2) if m_million_unit.group(2) else "개"
            return (val, f"count_absolute_{unit_suffix}")
        except ValueError: pass
    # 단위 없는 숫자 (음수 포함)
    try: return (float(text), "unitless_number") # float()는 "-0.08" 잘 처리
    except ValueError: pass
    return None

def extract_typed_values_for_fallback_pred(pred_text_input): # 모델 답변 내 모든 값 추출
    text = _normalize_text(pred_text_input)
    results = []
    current_pos = 0
    # 정규식에서 숫자 부분에 [-]? 추가하여 음수 지원
    patterns_ordered = [
        {'name': 'percent', 'regex': r"([-]?[\d.]+)\s*%", 
         'processor': lambda m: {'value': float(m.group(1)), 'type': 'percent', 'matched_text': m.group(0)} if m else None},
        {'name': 'monetary_jo', 'regex': r"([-]?[\d.]+)\s*(조원|조)", 
         'processor': lambda m: {'value': _parse_monetary_to_eokwon_fallback(m.group(1), "조"), 'type': 'monetary_eokwon', 'matched_text': m.group(0)} if m and _parse_monetary_to_eokwon_fallback(m.group(1), "조") is not None else None},
        {'name': 'monetary_eokwon', 'regex': r"([-]?[\d.]+)\s*(억원|억\s*원)", 
         'processor': lambda m: {'value': _parse_monetary_to_eokwon_fallback(m.group(1), "억원"), 'type': 'monetary_eokwon', 'matched_text': m.group(0)} if m and _parse_monetary_to_eokwon_fallback(m.group(1), "억원") is not None else None},
        {'name': 'monetary_eok', 'regex': r"([-]?[\d.]+)\s*억(?!\s*원)", 
         'processor': lambda m: {'value': _parse_monetary_to_eokwon_fallback(m.group(1), "억"), 'type': 'monetary_eokwon', 'matched_text': m.group(0)} if m and _parse_monetary_to_eokwon_fallback(m.group(1), "억") is not None else None},
        {'name': 'monetary_won', 'regex': r"([-]?[\d.]+)\s*원", 
         'processor': lambda m: {'value': _parse_monetary_to_eokwon_fallback(m.group(1), "원"), 'type': 'monetary_eokwon', 'matched_text': m.group(0)} if m and _parse_monetary_to_eokwon_fallback(m.group(1), "원") is not None else None},
        {'name': 'multiplier', 'regex': r"([-]?[\d.]+)\s*배",
         'processor': lambda m: {'value': float(m.group(1)), 'type': 'multiplier', 'matched_text': m.group(0)} if m else None},
        {'name': 'count_million', 'regex': r"([-]?[\d.]+)\s*백만\s*(\w+)?",
         'processor': lambda m: {'value': float(m.group(1)) * 1000000, 'type': f"count_absolute_{m.group(2) if m.group(2) else '개'}", 'matched_text': m.group(0)} if m else None},
        {'name': 'unitless_number', 'regex': r"([-]?[\d.]+)(?!\s*(?:년|월|일|분기|차|회|배|프로|퍼센트|%|[a-zA-Z]))",
         'processor': lambda m: {'value': float(m.group(1)), 'type': 'unitless_number', 'matched_text': m.group(1)} if m else None},
    ]
    while current_pos < len(text):
        sub_text = text[current_pos:]
        match_advanced_position = False
        for p_info in patterns_ordered:
            m = re.match(p_info['regex'], sub_text)
            if m:
                processed_item = None
                try: processed_item = p_info['processor'](m)
                except (ValueError, TypeError, IndexError): pass
                if processed_item:
                    results.append(processed_item)
                    current_pos += m.end()
                    match_advanced_position = True
                    break 
        if not match_advanced_position: current_pos += 1
    final_results = []; seen = set()
    for r in results:
        item_tuple = (r['value'], r['type']) # 중복 제거시 값과 유형이 모두 같아야 함
        if item_tuple not in seen: final_results.append(r); seen.add(item_tuple)
    return final_results

def compare_typed_values_with_tolerance(gt_val, pred_val, gt_type, pred_type, tolerance=0.05, abs_tolerance_for_zero=1e-9):
    # (이 함수는 이전과 동일 - 음수 값 자체는 float으로 잘 처리됨)
    original_gt_val, original_pred_val = gt_val, pred_val 

    if gt_type == "percent" and pred_type == "unitless_number": pred_val *= 100.0 
    elif gt_type == "unitless_number" and pred_type == "percent": pred_val /= 100.0
    elif gt_type == "monetary_eokwon" and pred_type == "unitless_number": pass
    elif gt_type == "unitless_number" and pred_type == "monetary_eokwon": pass
    elif gt_type.startswith("count_absolute_") and pred_type == "unitless_number": pass
    elif gt_type == "unitless_number" and pred_type.startswith("count_absolute_"): pass
    elif gt_type == "multiplier" and pred_type == "unitless_number": pass
    elif gt_type == "unitless_number" and pred_type == "multiplier": pass
    elif gt_type != pred_type : return False

    if abs(gt_val) < abs_tolerance_for_zero: return abs(pred_val) < abs_tolerance_for_zero
    # if abs(gt_val) == 0: return False # 위에서 abs_tolerance_for_zero 로 처리됨
    
    # gt_val이 0이면서 abs_tolerance_for_zero 보다 큰 경우는 없음. gt_val이 0이면 위에서 return.
    # 따라서 여기서 gt_val은 0이 아님 (또는 0에 매우 가깝지 않음).
    if abs(gt_val) < 1e-12 : # 매우 작은 수로 나누는 것 방지 (사실상 0인데 위에서 안 걸린 경우)
        return abs(gt_val - pred_val) < abs_tolerance_for_zero # 절대값 비교로 전환
        
    relative_error = abs(gt_val - pred_val) / abs(gt_val)
    return relative_error < tolerance

# --- 정확도 계산 메인 함수 (이전과 동일) ---
def calculate_accuracy_with_all_fallbacks(excel_path, 
                                           question_col_for_log,
                                           expected_answer_col, 
                                           generated_answer_col, 
                                           tolerance=0.05):
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError: print(f"🚨 오류: 파일을 찾을 수 없습니다 - {excel_path}"); return None
    required_cols = [expected_answer_col, generated_answer_col]
    if question_col_for_log: required_cols.append(question_col_for_log)
    for col_name in required_cols:
        if col_name not in df.columns: print(f"🚨 오류: '{col_name}' 열을 찾을 수 없습니다."); return None

    match_count = 0
    total_rows_compared = 0
    gt_extract_fail_count_simple = 0
    fallback_gt_parse_fail_count = 0

    print("\n--- 최종 폴백 포함 비교 로그 ---")
    for index, row in df.iterrows():
        question_text = str(row[question_col_for_log]) if question_col_for_log and pd.notna(row[question_col_for_log]) else "[질문 없음]"
        expected_value_text_raw = row[expected_answer_col]
        pred_text_raw = row[generated_answer_col]

        print(f"\n행 {index+2}: 질문='{question_text}'")
        print(f"  기대 정답 값 원본='{expected_value_text_raw}', 모델 답변 원본='{pred_text_raw}'")

        if pd.notna(expected_value_text_raw) and pd.notna(pred_text_raw):
            total_rows_compared += 1
            normalized_pred_text = str(pred_text_raw).replace(",", "") # 단순 비교시 사용
            row_match_found = False

            # 레벨 1 & 2: 단순 숫자 문자열 포함 비교
            gt_number_raw_extracted_simple = extract_key_number_from_gt_simple(str(expected_value_text_raw))
            
            final_search_terms_simple_for_log = "N/A (단순 GT 추출 실패)" # 로그용 변수 초기화
            if gt_number_raw_extracted_simple and gt_number_raw_extracted_simple.strip():
                print(f"  단순 비교 GT 추출(쉼표제거): '{gt_number_raw_extracted_simple}'")
                search_terms_set_simple = set()
                search_terms_set_simple.add(gt_number_raw_extracted_simple)
                try:
                    num_as_float = float(gt_number_raw_extracted_simple)
                    canonical_float_form = str(num_as_float)
                    if '.' in canonical_float_form and canonical_float_form.endswith('.0'):
                        canonical_float_form = canonical_float_form[:-2]
                    search_terms_set_simple.add(canonical_float_form)
                except ValueError: pass
                search_terms_set_simple.add(normalize_gt_number_trailing_zeros_simple(gt_number_raw_extracted_simple))
                
                final_search_terms_simple = {term for term in search_terms_set_simple if term and term.strip()}
                final_search_terms_simple_for_log = sorted(list(final_search_terms_simple), key=len, reverse=True)
                print(f"    단순 비교 검색 후보: {final_search_terms_simple_for_log}")

                for term_to_search in final_search_terms_simple_for_log: # 정렬된 순서대로 검색
                    if term_to_search in normalized_pred_text:
                        print(f"    ✅ 매칭 성공 (단순 문자열): 모델 답변 내에 '{term_to_search}' 발견")
                        row_match_found = True
                        break
            else:
                gt_extract_fail_count_simple += 1
                print(f"   ⚠️ 단순 비교 GT 숫자 추출 실패: '{expected_value_text_raw}'")

            # 레벨 3: 단위 고려 숫자 값 비교 (폴백)
            if not row_match_found:
                print(f"    단순 문자열 매칭 실패. 단위 고려 폴백 비교 시도...")
                parsed_gt_fallback = parse_typed_value_for_fallback_gt(str(expected_value_text_raw))
                
                if parsed_gt_fallback:
                    gt_val_fb, gt_type_fb = parsed_gt_fallback
                    print(f"      폴백 GT 파싱됨: 값={gt_val_fb}, 유형='{gt_type_fb}'")
                    all_pred_values_fb = extract_typed_values_for_fallback_pred(str(pred_text_raw))
                    
                    if not all_pred_values_fb:
                        print(f"      폴백 Pred에서 추출된 값 없음.")
                    else:
                        print(f"      폴백 Pred에서 추출된 값들 (중복제거됨): {all_pred_values_fb}")

                    for pred_item_fb in all_pred_values_fb:
                        pred_val_fb, pred_type_fb = pred_item_fb['value'], pred_item_fb['type']
                        if compare_typed_values_with_tolerance(gt_val_fb, pred_val_fb, gt_type_fb, pred_type_fb, tolerance):
                            print(f"    ✅ 매칭 성공 (폴백 - 단위고려): Pred 값={pred_val_fb} (유형='{pred_type_fb}')이 GT({gt_val_fb}, {gt_type_fb})와 일치")
                            row_match_found = True
                            break
                else:
                    fallback_gt_parse_fail_count += 1
                    print(f"    ❌ 폴백 GT 파싱 실패: '{expected_value_text_raw}' (단위 고려 파싱 불가)")
            
            if row_match_found:
                match_count += 1
            else:
                print(f"    ❌ 최종 매칭 실패 (단순 비교 시도: '{final_search_terms_simple_for_log}', 폴백 비교 결과 반영됨)")
        else:
            print(f"   기대 정답 값 또는 모델 답변이 비어있어 비교 제외.")

    if total_rows_compared == 0: print("⚠️ 경고: 비교할 수 있는 유효한 데이터 행이 없습니다."); return 0.0
    accuracy = match_count / total_rows_compared
    
    print(f"\n--- 최종 정확도 계산 결과 (모든 폴백 포함) ---")
    print(f"비교 대상 파일: {excel_path}")
    print(f"질문 열 (참고용): '{question_col_for_log}'")
    print(f"기대 정답 값 열: '{expected_answer_col}'")
    print(f"모델 답변 열: '{generated_answer_col}'")
    print(f"총 비교 시도된 행 수 (결측치 제외): {total_rows_compared}")
    if gt_extract_fail_count_simple > 0: print(f"  - 단순 GT 숫자 추출 실패 수: {gt_extract_fail_count_simple}")
    if fallback_gt_parse_fail_count > 0: print(f"  - 폴백 GT 파싱 실패 수 (단위고려): {fallback_gt_parse_fail_count}")
    print(f"조건 만족하는 행 수 (정답 처리): {match_count}")
    print(f"최종 정확도 (모든 폴백 포함): {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy

# --- 메인 실행 ---
if __name__ == "__main__":
    accuracy = calculate_accuracy_with_all_fallbacks(
        EXCEL_FILE_PATH,
        QUESTION_COLUMN_NAME_FOR_LOGGING, 
        EXPECTED_ANSWER_VALUE_COLUMN_NAME,
        GENERATED_ANSWER_COLUMN_NAME,
        tolerance=0.05
    )
    if accuracy is not None: print("\n계산이 완료되었습니다.")