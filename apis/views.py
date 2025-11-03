import re
import pandas as pd
from django.http import JsonResponse
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

# ── LLM (프롬프트/키 그대로)
llm_consistent = ChatAnthropic(
    model='claude-haiku-4-5',
    anthropic_api_key="",
    temperature=0,
    max_tokens=1000,
)

SQL_REGEX     = re.compile(r'"sql"\s*:\s*"([^"]+)"', re.IGNORECASE | re.DOTALL)
OPINION_REGEX = re.compile(r'"opinion"\s*:\s*"([^"]*)"', re.IGNORECASE | re.DOTALL)
MAIN_REGEX    = re.compile(r'"main"\s*:\s*"([^"]*)"', re.IGNORECASE | re.DOTALL)
SUB_REGEX     = re.compile(r'"sub"\s*:\s*"([^"]*)"', re.IGNORECASE | re.DOTALL)

def embed_kure(text: str) -> list[float]:
    emb = OpenAIEmbeddings(
        api_key="",
        model="text-embedding-3-small"
    )
    return emb.embed_query(text)

def filter_by_category_mean(op_vec, main: str, sub: str, candidate_ids=None):
    uid_filter = ""
    params = [main, sub]
    if candidate_ids:
        uid_filter = "AND uid = ANY(%s)"
        params.append(candidate_ids)
    sql = f"""
    WITH cat AS (
        SELECT uid, vec
        FROM kure_item_embeddings_v2
        WHERE main = %s AND sub = %s
        {uid_filter}
    ),
    sims AS (
        SELECT uid, 1.0 - (vec <=> %s) AS sim
        FROM cat
    )
    SELECT uid
    FROM sims
    WHERE sim >= (SELECT AVG(sim) FROM sims)
    ORDER BY sim DESC;
    """
    params.append(op_vec)
    with connection.cursor() as cur:
        cur.execute(sql, params)
        return [r[0] for r in cur.fetchall()]

# ✅ JSON 응답용 API
@csrf_exempt
def search_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 지원됩니다."}, status=405)

    try:
        import json
        body = json.loads(request.body or "{}")
        user_input = (body.get("query") or "").strip()
    except Exception:
        return JsonResponse({"error": "JSON 요청 파싱 실패"}, status=400)

    if not user_input:
        return JsonResponse({"error": "질문을 입력해 주세요."}, status=400)

    # (임시) RAG 컨텍스트 비활성화
    retrieved_docs = ""

    # ── LLM 질의
    message = f"""
        당신의 역할은 사용자의 자연어 입력을 해석하여 두 가지 결과를 동시에 생성하는 것입니다.

        1. 메타데이터(meta query)
        사람의 프로필 속성(성별, 출생년도/나이, 혼인, 학력, 직업, 가족규모, 자녀수, 소득, 흡연, 소유물, 지역/세부지역 등)을 정규화하고,
        이 값들만으로 RDB 질의를 위한 WHERE 절을 구성합니다.
        출력은 반드시 SELECT * FROM panel_records WHERE ...; 전체 문장으로 생성합니다.

        2. 오피니언(opinion) 
        사용자의 생각/성향/선호/감정·심리를 자연어 한 줄로 요약합니다.
        오피니언이 존재하면, 아래 메인/서브 카테고리 중 각각 1개를 선택합니다.
        [해시태그]
        #main = main 해시태그 / - = sub 해시태그
        #main 여가와 문화
        - 여행 이외의 모든 오프라인 문화생활
        - 여행 기반 오프라인 문화생활
        #main 일상요소
        - 경험 추억 등 과거와 관련된 행동 
        - 환경과 관련된 행동 
        - 일상적으로 반복하는 행동 
        #main 스타일 외모
        - 패션 관련 뷰티
        - 패션 외적인 뷰티
        #main 기술 및 정보
        - 디지털 도구 활용 
        #main 소비와 재정
        - 소비를 통해 이득을 취하는 경우
        - 소비를 통해 가치관을 표현 
        #main 건강 웰빙
        - 신체적 건강
        - 정신적, 심적인 건강 

        [생성규칙]
        A. SQL 생성
        메타 필드 중 값이 있는 것만 조건으로 연결 (AND) 
        <조건목록>
        *birth와 nchild는 int, 다른 조건 칼럼들은 string
        gender,	birth,	region,	subregion,	married, nchild, famsize,
        education_level,	job,	work,	p_income,	h_income,	owned_products,	phone_brand,
        phone_model,	car_ownship,	car_manufacturer,	car_model,	ever_smoked,	brand_smoked,	brand_smoked_ETC,
        ever_esmoked,	ever_smoked_brand_ETC,	ever_alcohol,	ever_smoked_ETC,	p_company

        B. 오피니언 존재 판정
        사용자의 선호/의견/감정/가치/취향/습관/루틴/습관/빈도/행동 의도가 드러나면 존재로 본다.
        예: “조용한 카페 선호”, “중고거래로 아끼는 편”, “요가로 스트레스 푼다”
        단순 사실(“서울에 산다”, “20대다”, “회사원이다”)만 있으면 부재
        존재하면 text에 문장 1개로 요약(user_input의 값을 최대한 반영하여 키워드를 살리기 군더더기 금지),
        동시에 가장 유사한 해시태그 메인 1개 + 서브 1개 선택
        오피니언 부재시 null로 처리하며 해시태그 main,sub 둘다 null로 처리

        [출력규칙]
        "sql","opinion","main","sub" 외에는 출력하지 않는다 선정이유와 LLM이용 과정을 출력하지 않는다
        또한 '''json형태로도 출력하지 않는다
        [출력예시 1]
        user_input: "서울 사는 대학생 중, 흡연을 하지 않고 환경문제에 관심이 많은 사람”
        "sql": "SELECT * FROM panel_records WHERE region = '서울' AND job = '대학생/대학원생' AND ever_smoked = '담배를 피워본 적이 없다';",
        "opinion": "환경문제에 관심이 많다"
        "main" : "일상요소"
        "sub" : "환경과 관련된 행동"
        [출력예시 2]
        user_input: "결혼을 하고 아이가 있는 돈을 아끼고 싶어하는 사람"
        "sql": "SELECT * FROM panel_records WHERE marital_status = '기혼' AND n_children IS NOT NULL;",
        "opinion": "돈을 아끼고 싶어한다"
        "main" : "소비와 재정"
        "sub" : "소비를 통해 이득을 취하는 경우"
        [출력예시 3]
        user_input: "아이폰을 사용하는 중년"
        "sql": "SELECT * FROM panel_records WHERE phone_model LIKE '%아이폰%' AND birth_year BETWEEN 1976 AND 1990;",
        "opinion": "null"
        "main" : "null"
        "sub" : "null"
        {user_input}

        참고: 
        {retrieved_docs}

    """.strip()

    try:
        resp = llm_consistent.invoke([HumanMessage(content=message)])
        content = getattr(resp, "content", "").strip()
    except Exception as e:
        return JsonResponse({"error": f"LLM 오류: {str(e)}"}, status=500)

    # ── LLM 결과 파싱
    m_sql  = SQL_REGEX.search(content)
    m_op   = OPINION_REGEX.search(content)
    m_main = MAIN_REGEX.search(content)
    m_sub  = SUB_REGEX.search(content)

    if not m_sql:
        return JsonResponse({
            "error": 'LLM 응답에서 "sql" 항목을 찾지 못했습니다.',
            "llm_output": content
        }, status=500)

    sql_text = m_sql.group(1).strip().rstrip(";")
    opinion = (m_op.group(1).strip() if m_op else "") or ""
    main    = (m_main.group(1).strip() if m_main else "") or ""
    sub     = (m_sub.group(1).strip() if m_sub else "") or ""

    # ── RDB 검색
    try:
        with connection.cursor() as cur:
            cur.execute(sql_text)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []
    except Exception as e:
        return JsonResponse({
            "error": f"SQL 실행 오류: {e}",
            "sql_text": sql_text
        }, status=500)

    # ── 결과 JSON 반환
    return JsonResponse({
        "last_query": user_input,
        "sql_text": sql_text,
        "opinion": opinion or None,
        "main": main or None,
        "sub": sub or None,
        "count": len(rows),
        "columns": cols,
        "rows": rows[:10],
    }, json_dumps_params={"ensure_ascii": False})
