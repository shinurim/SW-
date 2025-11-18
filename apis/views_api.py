import os, re, json
from django.http import JsonResponse
from django.db import connections
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# LLM OUTPUT값을 반환해 주는 함수⭐⭐⭐⭐⭐⭐⭐⭐
KURE_MODEL_PATH = os.getenv("KURE_MODEL_PATH", "nlpai-lab/KURE-v1")
KURE_NORMALIZE  = os.getenv("KURE_NORMALIZE", "false").lower() in ("1", "true", "yes")
DOCVEC_VEC_COL  = "embedding"   # ← insight_docvec 컬럼명 확정

# ===================== KURE 임베딩 =====================
_sentence_model = None
def _get_kure_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(KURE_MODEL_PATH, device="cpu")
    return _sentence_model

def _kure_embed(text: str) -> list[float]:
    model = _get_kure_model()
    vec = model.encode([text], normalize_embeddings=KURE_NORMALIZE)[0]
    return [float(x) for x in vec.tolist()]

def _as_vector_param(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

# ===================== DB 헬퍼 =====================
def _dictfetchall(cur):
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def _nullish(v) -> bool:
    return v is None or str(v).strip().lower() in ("","-", "null", "none")

# ===================== 문서 검색 (KURE + pgvector) =====================
def _retrieve_docs_from_insight(query_text: str, k: int = 5):
    """
    insight_docvec.embedding 기반 코사인 유사도 검색 → LLM 참고용 문맥 반환
    """
    try:
        qv = _kure_embed(query_text)
        qv_param = _as_vector_param(qv)
    except Exception as e:
        return {"error": f"KURE 임베딩 실패: {e}", "retrieved_docs": [], "retrieved_block": ""}

    sql = f"""
        SELECT id, content, 1.0 - ({DOCVEC_VEC_COL} <=> %s::vector) AS score
        FROM insight_docvec
        WHERE content IS NOT NULL
        ORDER BY {DOCVEC_VEC_COL} <=> %s::vector ASC
        LIMIT %s;
    """

    try:
        with connections["vecdb"].cursor() as cur:
            cur.execute(sql, [qv_param, qv_param, int(k)])
            rows = cur.fetchall()
    except Exception as e:
        return {"error": f"insight_docvec 검색 실패: {e}", "retrieved_docs": [], "retrieved_block": ""}

    docs = []
    for rid, content, score in rows:
        text = (content or "").strip()
        if len(text) > 800:
            text = text[:800] + " ..."
        docs.append({"id": str(rid), "score": round(score or 0, 4), "content": text})

    block = "\n\n".join(f"[{i+1}] {d['content']}" for i, d in enumerate(docs))
    return {"retrieved_docs": docs, "retrieved_block": block}

# ===================== LLM 초기화 =====================
llm_consistent = ChatAnthropic(
    model="claude-opus-4-20250514", #claude-haiku-4-5
    anthropic_api_key=key,
    temperature=0,
    max_tokens=1000,
)

# ===================== 정규식 =====================
SQL_REGEX     = re.compile(r'"?sql"?\s*:\s*"?(SELECT[^"\n]+)"?', re.IGNORECASE | re.DOTALL)
OPINION_REGEX = re.compile(r'"?opinion"?\s*:\s*"?(.*?)"?\s*(?:\n|$)', re.IGNORECASE | re.DOTALL)
MAIN_REGEX    = re.compile(r'"?main"?\s*:\s*"?(.*?)"?\s*(?:\n|$)', re.IGNORECASE | re.DOTALL)
SUB_REGEX     = re.compile(r'"?sub"?\s*:\s*"?(.*?)"?\s*(?:\n|$)', re.IGNORECASE | re.DOTALL)

# ===================== API =====================
def run_stage1_nl_to_meta(user_input: str) -> dict:

    user_input = (user_input or "").strip()
    if not user_input:
        raise ValueError("질문이 비어 있습니다.")

    # 2️⃣ 문서 참조 (KURE + pgvector)
    retr = _retrieve_docs_from_insight(user_input, k=2)
    retrieved_block = retr.get("retrieved_block", "")
    retrieved_docs_list = retr.get("retrieved_docs", [])

    # ✅ 추가: 콘솔에서 참조 문서 내용 확인
    print("🔍 Retrieved Block Preview:")
    print("──────────────────────────────")
    print(retrieved_block[:800])  # 상위 800자만 미리보기
    print("──────────────────────────────")

    # 3️⃣ 프롬프트 구성
    message = f"""
        강제규칙:[출력규칙][출력예시]를 반드시 지킨다 
        당신의 역할은 사용자의 자연어 입력을 해석하여 두 가지 결과를 동시에 생성하는 것입니다.

        1. 메타데이터(meta query)
        사람의 프로필 속성(성별, 출생년도/나이, 혼인, 학력, 직업, 가족규모, 자녀수, 소득, 흡연, 소유물, 지역/세부지역 등)을 정규화하고,
        이 값들만으로 RDB 질의를 위한 WHERE 절을 구성합니다.
        출력은 반드시 SELECT * FROM panel_records WHERE ...; 전체 문장으로 생성합니다.
        조건이 없을 경우 SELECT * FROM panel_records;

        2. 오피니언(opinion) 
        사용자의 생각/성향/선호/감정·심리를 자연어 한 줄로 요약합니다.
        오피니언이 존재하면, 아래 메인/서브 카테고리 중 각각 1개를 선택합니다.
        [해시태그]
        main과 sub는 반드시 아래 목록에 나온 문구를 그대로 사용하며,
        다른 단어를 붙이거나 순서를 바꾸거나 변형하지 않는다.
        #main = main 해시태그 / - = sub 해시태그
        #main "여가와 문화"
        - "여행 이외의 모든 오프라인 문화생활"
        - "여행 기반 오프라인 문화생활"
        #main "일상 요소"
        - "경험 추억 등 과거와 관련된 행동" 
        - "환경과 관련된 행동" 
        - "일상적으로 반복하는 행동" 
        #main "스타일 외모"
        - "패션 관련 뷰티"
        - "패션 외적인 뷰티"
        #main "기술 및 정보"
        - "디지털 도구 활용" 
        #main "소비와 재정"
        - "소비를 통해 이득을 취하는 경우"
        - "소비를 통해 가치관을 표현" 
        #main "건강 웰빙"
        - "신체적 건강"
        - "신체적·심적인 건강" 

        [생성규칙]
        A. SQL 생성
        메타 필드 중 값이 있는 것만 조건으로 연결 (AND) 
        형식 준수: SQL WHERE 절의 모든 문자열 값은 반드시 작은 따옴표(')로 묶어야 한다
        <조건목록>
        *birth와 nchild는 int, 다른 조건 칼럼들은 string
        gender, birth,  region, subregion,  married, nchild, famsize,
        education_level,    job,    work,   p_income,   h_income,   owned_products, phone_brand,
        phone_model,      car_ownship,      car_manufacturer,   car_model,    ever_smoked,      brand_smoked,   brand_smoked_ETC,
        ever_esmoked,   ever_smoked_brand_ETC,  ever_alcohol,   ever_smoked_ETC,      p_company

        B. 오피니언 존재 판정
        사용자의 선호/의견/감정/가치/취향/습관/루틴/습관/빈도/행동 의도가 드러나면 존재로 본다.
        예: “조용한 카페 선호”, “중고거래로 아끼는 편”, “요가로 스트레스 푼다”
        단순 사실(“서울에 산다”, “20대다”, “회사원이다”)만 있으면 부재
        존재하면 text에 문장 1개로 요약(user_input의 값을 최대한 반영하여 키워드를 살리기 군더더기 금지),
        동시에 가장 유사한 해시태그 메인 1개 + 서브 1개 선택
        오피니언 부재시 "-"로 처리하며 해시태그 main,sub 둘다 "-"로 처리            
        
        [출력규칙]
        "sql","opinion","main","sub" 외에는 출력하지 않는다 선정이유와 LLM 연산 과정을 출력하지 않는다
        출력은 아래 4개 키만 포함해야 하며, 다른 문장/마크다운/설명은 절대 포함하지 않는다.  
        또한 '''json형태로도 출력하지 않는다
        부재시 "-"으로만 처리한다 

        [출력예시 1]
        user_input: "서울 사는 대학생 중, 흡연을 하지 않고 환경문제에 관심이 많은 사람”
        "sql": "SELECT * FROM panel_records WHERE region = '서울' AND job = '대학생/대학원생' AND ever_smoked = '담배를 피워본 적이 없다';",
        "opinion": "환경문제에 관심이 많다"
        "main" : "일상요소"
        "sub" : "환경과 관련된 행동"
        [출력예시 2]
        user_input: "결혼을 하고 아이가 있는 돈을 아끼고 싶어하는 사람"
        "sql": "SELECT * FROM panel_records WHERE married = '기혼' AND nchild IS NOT NULL;",
        "opinion": "돈을 아끼고 싶어한다"
        "main" : "소비와 재정"
        "sub" : "소비를 통해 이득을 취하는 경우"
        [출력예시 3]
        user_input: "아이폰을 사용하는 중년"
        "sql": "SELECT * FROM panel_records WHERE phone_model LIKE '%아이폰%' AND birth BETWEEN 1961 AND 1990;",
        "opinion": "-"
        "main" : "-"
        "sub" : "-"
        [출력예시 4]
        user_input: "대학생 또는 교직에 종사하는 사람 중, 디지털 도구 활용 능력이 뛰어나다고 생각하는 사람"
        "sql": "SELECT * FROM panel_records WHERE job IN ('대학생/대학원생', '교직 (교수, 교사, 강사 등)');",
        "opinion": "디지털 도구 활용 능력이 뛰어나다고 생각한다"
        "main": "기술 및 정보"
        "sub": "디지털 도구 활용"
        [출력예시 5]
        user_input: "IT 분야에 종사하며 취미로 캠핑을 즐기는 사람"
        "sql": "SELECT * FROM panel_records WHERE work = 'IT';",
        "opinion": "취미로 캠핑을 즐긴다"
        "main": "여가와 문화"
        "sub": "여행 기반 오프라인 문화생활"

        # 📌 사용자 입력
        {user_input}

        참고: 
        # a. 성별 (Gender): 반드시 남 = M / 여 = F 으로 이분법적으로 처리
        # b. 결혼 여부 (Married): ('기혼','미혼','기타(사별/이혼 등) 
        # c. 자녀 수 (nchild):** 자녀가 있다는 질문은 **nchild > 0** 또는 **nchild IS NOT NULL**을 사용합니다. 특정 자녀 수는 **nchild = [숫자]**를 사용하며, nchild는 **정수형(int)** 칼럼이므로 작은따옴표로 묶지 않습니다.
        # d. 가족 수 (famsize): '1명(혼자 거주)','2명','3명','4명','5명 이상' 중 선택 다가구의 경우 혼자 거주하는 경우만 제외한다
        {retrieved_block}
    """.strip()

    # 4️⃣ LLM 호출
    try:
        resp = llm_consistent.invoke([HumanMessage(content=message)])
        content = getattr(resp, "content", "").strip()
    except Exception as e:
        return {"error": f"LLM 오류: {str(e)}"}

    # 5️⃣ 파싱
    m_sql = SQL_REGEX.search(content)
    m_op  = OPINION_REGEX.search(content)
    m_ma  = MAIN_REGEX.search(content)
    m_su  = SUB_REGEX.search(content)

    if not m_sql:
        return {
            "error": 'LLM 응답에서 "sql" 항목을 찾지 못했습니다.',
            "llm_output": content[:800]
        }

    sql_text = m_sql.group(1).strip().rstrip(";")
    opinion  = (m_op.group(1).strip() if m_op else "")
    main     = (m_ma.group(1).strip() if m_ma else "")
    sub      = (m_su.group(1).strip() if m_su else "")

    # ✅ 6️⃣ 정규화: "-", "", "null" → None
    opinion_value = None if _nullish(opinion) else opinion
    main_value    = None if _nullish(main)    else main
    sub_value     = None if _nullish(sub)     else sub

    # ✅ 핵심: main/sub 중 하나라도 비면 opinion도 None으로 강제(= 2단계로)
    if opinion_value is not None and (main_value is None or sub_value is None):
        opinion_value = None

    # 7️⃣ 반환
    return {
        "sql_text": sql_text,
        "opinion": opinion_value,
        "main": main_value,
        "sub": sub_value,
        "retrieved_docs": retrieved_docs_list
    }
