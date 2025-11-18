# 라이브러리
import os, re, json, time, ast
import numpy as np

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db import connections
from sentence_transformers import SentenceTransformer

# (선택) 1단계 메타 함수: query -> {sql_text, opinion, main, sub}
try:
    from apis.views_api import run_stage1_nl_to_meta
except Exception as e:
    run_stage1_nl_to_meta = None

# ===========================
# 공통 유틸
# ===========================

def _dictfetchall(cur):
    cols = [c[0] for c in cur.description] if cur.description else []
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols


# SELECT ... FROM panel_records ... 에서 WHERE만 추출
_WHERE_RE = re.compile(
    r"select\s+\*\s+from\s+panel_records\s*(where\s+.+?)?\s*;?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_where(sql_text: str) -> str:
    s = (sql_text or "").strip()
    m = _WHERE_RE.search(s)
    if not m:
        return ""
    where = m.group(1) or ""
    # ORDER BY / LIMIT / OFFSET 제거
    where = re.split(r"\b(order\s+by|limit|offset)\b", where, flags=re.IGNORECASE)[0].strip()
    return where


_ALLOWED_COLS = {
    "id","gender","birth","region","subregion","married","nchild","famsize",
    "education_level","job","work","p_income","h_income",
    "owned_products","phone_brand","phone_model",
    "car_ownship","car_manufacturer","car_model",
    "ever_smoked","brand_smoked","brand_smoked_ETC",
    "ever_esmoked","ever_smoked_brand_ETC","ever_alcohol","p_company",
    "loyalty",  # 2단계 정렬에 필요
}
_COL_RE = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*"
    r"(?:=|<>|!=|>=|<=|>|<|in\s*\(|like\b|ilike\b|between\b|is\s+null|is\s+not\s+null)",
    flags=re.IGNORECASE
)


def _columns_from_where(where_sql: str):
    if not where_sql:
        return []
    cols = set()
    for m in _COL_RE.finditer(where_sql):
        c = m.group(1)
        if c.lower() in {"and","or","not","between","is","null"}:
            continue
        if c in _ALLOWED_COLS:
            cols.add(c)
    cols.add("id")
    return sorted(cols)


def _nullish(v) -> bool:
    return v is None or str(v).strip().lower() in ("", "null", "none", "-")


def _vendor_placeholder():
    vendor = connections["default"].vendor  # 'postgresql' | 'sqlite' | 'mysql' 등
    return ("?", "?") if vendor == "sqlite" else ("%s", "%s")


def _clean_tag(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return s
    s = re.sub(r'^[\'"\s]+|[\'"\s,;]+$', '', s)
    s = re.sub(r"\s+", " ", s)
    return s


def _split_qids(qids_used: str):
    if not qids_used:
        return []
    parts = [p.strip() for p in str(qids_used).split("|") if p.strip()]
    seen, unique = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _normalize_retrieved_block(retrieved_docs):
    """list/dict 어떤 형태로 와도 4단계에서 쓰기 편한 dict로 정규화"""
    if not retrieved_docs:
        return None
    if isinstance(retrieved_docs, dict):
        return retrieved_docs
    if isinstance(retrieved_docs, list):
        return retrieved_docs[0] if retrieved_docs else None
    return None


# ===========================
# KURE 설정 / 임베딩 유틸
# ===========================

KURE_MODEL_PATH = os.getenv("KURE_MODEL_PATH", "nlpai-lab/KURE-v1")
KURE_NORMALIZE  = os.getenv("KURE_NORMALIZE", "false").lower() in ("1", "true", "yes")

KURE_TABLE     = os.getenv("KURE_TABLE", "kure_item_embeddings_v2")
KURE_UID_COL   = os.getenv("KURE_UID_COL", "uid")
KURE_VEC_COL   = os.getenv("KURE_VEC_COL", "vec")
KURE_MAIN_COL  = os.getenv("KURE_MAIN_COL", "main")
KURE_SUB_COL   = os.getenv("KURE_SUB_COL",  "sub")
KURE_QIDS_COL  = os.getenv("KURE_QIDS_COL", "qids_used")

RDB_BASE_COLS = ["id","gender","birth","region","subregion"]

_sentence_model = None
def _get_kure_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            print(f"[INFO] Loading KURE model: {KURE_MODEL_PATH}")
            _sentence_model = SentenceTransformer(KURE_MODEL_PATH, device="cpu")
            print("[INFO] KURE model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"KURE 모델 로드 실패: {e}")
    return _sentence_model


def _kure_embed(text: str) -> list[float]:
    s = (text or "").strip()
    if not s:
        raise ValueError("빈 opinion 입니다.")
    model = _get_kure_model()
    vecs = model.encode([s], normalize_embeddings=KURE_NORMALIZE)
    if isinstance(vecs, np.ndarray):
        vec = vecs[0].tolist()
    else:
        vec = list(vecs[0])
    if not vec or len(vec) == 0:
        raise RuntimeError("KURE 임베딩 결과가 비었습니다.")
    return [float(x) for x in vec]


# ===========================
# 3단계 핵심 로직 (RDB + vecdb 교집합 + 유사도 정렬)
# ===========================

def _run_insight_core(
    *,
    sql_text: str,
    opinion: str,
    main: str,
    sub: str,
    where_sql: str,
    limit: int,
    offset: int,
    candidate_cap: int,
):
    """
    insight_filter의 핵심만 떼어낸 함수.
    반환: (total_count, page_rows(list[dict]), elapsed_str)
    """
    t0 = time.perf_counter()

    # 1️⃣ RDB 후보군 id 먼저 추출
    candidate_ids = None
    if where_sql:
        ids_sql = f"""
            SELECT id
            FROM panel_records
            {where_sql}
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """
        with connections["default"].cursor() as cur:
            cur.execute(ids_sql, [candidate_cap, 0])
            candidate_ids = tuple(str(r[0]) for r in cur.fetchall())

    if not candidate_ids:
        elapsed = time.perf_counter() - t0
        return 0, [], f"{elapsed:.2f} sec"

    # 2️⃣ vecdb에서 main, sub에 해당하면서 qids_used가 NULL이 아닌 UID + vec 가져오기
    qids_sql = f"""
        SELECT {KURE_UID_COL} AS uid, {KURE_QIDS_COL} AS qids, {KURE_VEC_COL} AS vec
        FROM {KURE_TABLE}
        WHERE {KURE_MAIN_COL} = %s
          AND {KURE_SUB_COL}  = %s
          AND {KURE_QIDS_COL} IS NOT NULL
    """
    with connections["vecdb"].cursor() as cur:
        cur.execute(qids_sql, [main, sub])
        vec_rows = cur.fetchall()  # (uid, qids_used, vec)

    if not vec_rows:
        elapsed = time.perf_counter() - t0
        return 0, [], f"{elapsed:.2f} sec"

    # 3️⃣ RDB 후보군과 uid 교집합
    candidate_set = set(candidate_ids)
    vec_filtered = [
        (str(uid), qids, vec) for uid, qids, vec in vec_rows
        if str(uid) in candidate_set
    ]
    if not vec_filtered:
        elapsed = time.perf_counter() - t0
        return 0, [], f"{elapsed:.2f} sec"

    # uid -> vec / qids 매핑 + 전체 q 컬럼 집합
    uid_to_vec = {}
    uid_to_qids = {}
    qid_union = set()

    for uid, qids, vec in vec_filtered:
        uid_to_vec[uid] = vec
        q_list = _split_qids(qids)
        uid_to_qids[uid] = q_list
        qid_union.update(q_list)

    qid_cols = sorted([q for q in qid_union if re.fullmatch(r"q\d+", q)])

    # 4️⃣ RDB에서 이 uid들에 대한 패널 정보 + q답변 조회
    where_cols = _columns_from_where(where_sql)
    select_cols = (where_cols or RDB_BASE_COLS) + qid_cols
    # 중복 제거
    seen, unique_cols = set(), []
    for c in select_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    select_list = ", ".join(unique_cols) if unique_cols else "id"

    rdb_sql = f"""
        SELECT {select_list}
        FROM panel_records
        WHERE id = ANY(%s::text[])
    """
    with connections["default"].cursor() as cur:
        cur.execute(rdb_sql, [list(uid_to_vec.keys())])
        rdb_rows = cur.fetchall()
        rdb_cols = [c[0] for c in cur.description] if cur.description else []

    # 5️⃣ RDB 기준으로 "qids_used에 해당하는 답이 하나도 없는 uid" 제거
    col_idx = {c: i for i, c in enumerate(rdb_cols)}
    rows_raw = {}      # uid -> 기본 row(dict)
    answers_map = {}   # uid -> {q: value}
    valid_uids = []    # 최종 유사도 계산 대상 uid

    for r in rdb_rows:
        d = {c: r[i] for c, i in col_idx.items()}
        uid = str(d.get("id"))
        q_all = uid_to_qids.get(uid, [])

        answers = {}
        for q in q_all:
            if q in d and d[q] is not None:
                answers[q] = d[q]

        if not answers:
            continue

        rows_raw[uid] = d
        answers_map[uid] = answers
        valid_uids.append(uid)

    if not valid_uids:
        elapsed = time.perf_counter() - t0
        return 0, [], f"{elapsed:.2f} sec"

    # 6️⃣ opinion 임베딩
    qv = _kure_embed(opinion)
    qv_np = np.array(qv, dtype=np.float32)
    qnorm = np.linalg.norm(qv_np) + 1e-8

    # 7️⃣ "답이 있는 uid"만 대상으로 vec 유사도 계산
    sim_list = []
    for uid in valid_uids:
        vec = uid_to_vec[uid]
        if isinstance(vec, str):
            try:
                vec_list = ast.literal_eval(vec)
            except Exception:
                continue
        else:
            vec_list = vec

        vec_np = np.array(vec_list, dtype=np.float32)
        vnorm = np.linalg.norm(vec_np) + 1e-8
        sim = float(np.dot(qv_np, vec_np) / (vnorm * qnorm))
        sim_list.append((uid, sim))

    if not sim_list:
        elapsed = time.perf_counter() - t0
        return 0, [], f"{elapsed:.2f} sec"

    # 8️⃣ 유사도 내림차순 정렬 + 페이지네이션
    sim_list.sort(key=lambda x: x[1], reverse=True)
    uid_ranked = [uid for uid, _ in sim_list]
    total_count = len(uid_ranked)

    uid_page = uid_ranked[offset: offset + limit]
    sim_map = {uid: sim for uid, sim in sim_list}

    # 9️⃣ 최종 rows_out 조립 (여기서 spec의 "data"로 사용)
    rows_out = []
    for uid in uid_page:
        base = rows_raw[uid].copy()
        base["qids_used"] = list(answers_map[uid].keys())
        base["answers"] = answers_map[uid]
        base["sim"] = sim_map[uid]
        rows_out.append(base)

    elapsed = time.perf_counter() - t0
    return total_count, rows_out, f"{elapsed:.2f} sec"


# ===========================
# 메인 엔드포인트: 2단계 + 3단계 자동 분기
# ===========================

@csrf_exempt
@require_http_methods(["POST"])
def rdb_gateway(request):

    # 공통 기본값
    sql_text = ""
    opinion = None
    main = None
    sub = None

    # 0) 요청 파싱
    try:
        body = json.loads(request.body or "{}")
    except Exception:
        return JsonResponse({"error": "JSON 파싱 실패"}, status=400)

    # 입력은 sql_text 또는 sql 둘 다 지원
    sql_text_in = (body.get("sql_text") or body.get("sql") or "").strip()
    # query는 자연어 검색용
    query_in = (body.get("query") or "").strip()

    limit  = int(body.get("limit") or 20)
    offset = int(body.get("offset") or 0)
    candidate_cap = int(body.get("candidate_cap") or 1000)

    retrieved_docs  = body.get("retrieved_docs")
    retrieved_block = body.get("retrieved_block") or _normalize_retrieved_block(retrieved_docs)

    # 1) sql_text / sql 이 오면 그대로 사용 (/search/sql 용)
    if sql_text_in:
        sql_text = sql_text_in
        # opinion/main/sub를 옵션으로 함께 받을 수도 있음 (수동 수정 케이스)
        opinion = body.get("opinion")
        main = body.get("main")
        sub = body.get("sub")

    # 2) query만 오면 1단계 메타 함수(run_stage1_nl_to_meta) 호출 (/search/text 용)
    elif query_in:
        if run_stage1_nl_to_meta is None:
            return JsonResponse(
                {"error": "메타 생성 함수(run_stage1_nl_to_meta)가 설정되어 있지 않습니다."},
                status=500,
            )

        try:
            meta = run_stage1_nl_to_meta(query_in)
        except ValueError as e:
            # user_input 비었을 때 등
            return JsonResponse({"error": str(e)}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"메타 생성 호출 오류: {e}"}, status=500)

        sql_text = (meta.get("sql_text") or "").strip()
        opinion = meta.get("opinion")
        main = meta.get("main")
        sub = meta.get("sub")

        if not sql_text:
            return JsonResponse(
                {"error": "메타 생성 결과에 sql_text가 없습니다.", "meta": meta},
                status=500,
            )
    else:
        # 둘 다 없으면 오류
        return JsonResponse({"error": "sql 또는 query 중 하나가 필요합니다."}, status=400)

    # opinion/main/sub 정규화
    opinion_norm = None if _nullish(opinion) else opinion
    main_norm = None if _nullish(main) else _clean_tag(main)
    sub_norm = None if _nullish(sub) else _clean_tag(sub)

    # WHERE 추출
    where_sql = _extract_where(sql_text)

    # 🔸 3단계 사용 여부: opinion 있고 main/sub도 있어야 함
    use_insight = bool(opinion_norm and main_norm and sub_norm)

    # ===========================
    # 3단계: opinion 기반 insight 필터
    # ===========================
    if use_insight:
        try:
            total, data_rows, elapsed_str = _run_insight_core(
                sql_text=sql_text,
                opinion=opinion_norm,
                main=main_norm,
                sub=sub_norm,
                where_sql=where_sql,
                limit=limit,
                offset=offset,
                candidate_cap=candidate_cap,
            )
        except Exception as e:
            return JsonResponse(
                {"error": f"Insight 처리 중 오류: {type(e).__name__}: {e}"},
                status=500,
            )

        return JsonResponse(
            {
                "sql": sql_text,
                "opinion": opinion_norm,
                "main": main_norm,
                "sub": sub_norm,
                "count": int(total),
                "sql_executed_time": elapsed_str,
                "data": data_rows,
                "retrieved_block": retrieved_block, # 참조자료
            },
            json_dumps_params={"ensure_ascii": False},
        )

    # ===========================
    # 2단계: 순수 RDB 검색 (기존 rdb_gateway)
    # ===========================

    where_clause = f" {where_sql}" if where_sql else ""

    # 기본 SELECT 컬럼
    select_cols = _columns_from_where(where_sql) or ["id", "gender", "birth", "region", "subregion"]

    # loyalty 기준 정렬이므로 loyalty 컬럼도 포함되게 보장
    if "loyalty" not in select_cols:
        select_cols.append("loyalty")

    select_list = ", ".join(select_cols)
    lim_ph, off_ph = _vendor_placeholder()

    # loyalty 기준 정렬
    order_by_clause = "ORDER BY loyalty DESC, id DESC"

    page_sql = f"""
        SELECT {select_list}
        FROM panel_records
        {where_clause}
        {order_by_clause}
        LIMIT {lim_ph} OFFSET {off_ph}
    """.strip()

    count_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM panel_records
        {where_clause}
    """.strip()

    try:
        t0 = time.perf_counter()

        with connections["default"].cursor() as cur:
            cur.execute(page_sql, [int(limit), int(offset)])
            rows, cols = _dictfetchall(cur)

        with connections["default"].cursor() as cur:
            cur.execute(count_sql)
            total = cur.fetchone()[0]

        elapsed = time.perf_counter() - t0
        sql_executed_time = f"{elapsed:.2f} sec"

        # SQL 직접 실행(/search/sql)인데 opinion 안 온 경우 → 명세서처럼 N/A 세팅
        if sql_text_in and not query_in and opinion_norm is None:
            opinion_out = "N/A (User-provided SQL)"
            main_out = "N/A"
            sub_out = "N/A"
        else:
            opinion_out = opinion_norm
            main_out = main_norm
            sub_out = sub_norm

        # 결과 없음도 스키마 유지
        if not rows or total == 0:
            return JsonResponse(
                {
                    "sql": sql_text,
                    "opinion": opinion_out,
                    "main": main_out,
                    "sub": sub_out,
                    "count": 0,
                    "sql_executed_time": sql_executed_time,
                    "data": [],
                },
                json_dumps_params={"ensure_ascii": False},
            )

        # 결과 있음
        return JsonResponse(
            {
                "sql": sql_text,
                "opinion": opinion_out,
                "main": main_out,
                "sub": sub_out,
                "count": int(total),
                "sql_executed_time": sql_executed_time,
                "data": rows,
            },
            json_dumps_params={"ensure_ascii": False},
        )

    except Exception as e:
        # IndexError는 '결과 없음'으로 처리
        if isinstance(e, IndexError):
            return JsonResponse(
                {
                    "sql": sql_text,
                    "opinion": opinion_norm,
                    "main": main_norm,
                    "sub": sub_norm,
                    "count": 0,
                    "sql_executed_time": "0.00 sec",
                    "data": [],
                    "message": "결과 없음 (IndexError)",
                },
                json_dumps_params={"ensure_ascii": False},
            )

        # 그 외 에러는 에러 메시지 반환
        return JsonResponse(
            {
                "error": f"RDB 실행 오류: {type(e).__name__}: {e}",
                "sql": sql_text,
                "where": where_sql,
                "select_cols": select_cols,
                "db_vendor": connections["default"].vendor,
            },
            status=500,
        )
