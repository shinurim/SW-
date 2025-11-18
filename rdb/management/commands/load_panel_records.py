import os
import math
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from rdb.models import PanelRecord

META_FIELDS = [
    "id", "gender", "birth", "region", "subregion", "married", "nchild",
    "famsize", "education_level", "job", "work", "p_income", "h_income",
    "owned_products", "phone_brand", "phone_model", "car_ownship",
    "car_manufacturer", "car_model", "ever_smoked", "brand_smoked",
    "brand_smoked_ETC", "ever_esmoked", "ever_alcohol",
    "ever_smoked_brand_ETC", "ever_smoked_ETC", "p_company", "loyalty",
]

QUESTION_MAP = {
    "q1":  "여러분이 여름철 물놀이 장소로 가장 선호하는 곳은 어디입니까?",
    "q2":  "여러분의 휴대폰 갤러리에 가장 많이 저장되어져 있는 사진은 무엇인가요?",
    "q3":  "갑작스런 비로 우산이 없을 때 여러분은 어떻게 하시나요?",
    "q4":  "여러분이 절대 포기할 수 없는 여름 패션 필수템은 무엇인가요?",
    "q5":  "여러분은 평소 개인정보보호를 위해 어떤 습관이 있으신가요?",
    "q6":  "여러분은 초콜릿을 주로 언제 드시나요?",
    "q7":  "여러분은 할인, 캐시백, 멤버십 등 포인트 적립 혜택을 얼마나 신경 쓰시나요?",
    "q8":  "평소 일회용 비닐봉투 사용을 줄이기 위해 어떤 노력을 하고 계신가요?",
    "q9":  "여러분은 여행갈 때 어떤 스타일에 더 가까우신가요?",
    "q10": "여러분은 본인을 미니멀리스트와 맥시멀리스트 중 어디에 더 가깝다고 생각하시나요?",
    "q11": "여러분은 요즘 어떤 분야에서 AI 서비스를 활용하고 계신가요?",
    "q12": "여러분은 최근 가장 지출을 많이 한 곳은 어디입니까?",
    "q13": "여러분의 여름철 최애 간식은 무엇인가요?",
    "q14": "여러분은 야식을 먹을 때 보통 어떤 방법으로 드시나요?",
    "q15": "여러분이 지금까지 해본 다이어트 중 가장 효과 있었던 방법은 무엇인가요?",
    "q16": "여름철 땀 때문에 겪는 불편함은 어떤 것이 있는지 모두 선택해주세요.",
    "q17": "여러분이 가장 중요하다고 생각하는 행복한 노년의 조건은 무엇인가요?",
    "q18": "여러분은 외부 식당에서 혼자 식사하는 빈도는 어느 정도인가요?",
    "q19": "여러분은 아침에 기상하기 위해 어떤 방식으로 알람을 설정해두시나요?",
    "q20": "여러분은 버리기 아까운 물건이 있을 때, 주로 어떻게 하시나요?",
    "q21": "여러분은 다가오는 여름철 가장 걱정되는 점이 무엇인가요?",
    "q22": "빠른 배송(당일·새벽·직진 배송) 서비스를 주로 어떤 제품을 구매할 때 이용하시나요?",
    "q23": "여러분은 올해 해외여행을 간다면 어디로 가고 싶나요? 모두 선택해주세요.",
    "q24": "여러분이 사용해 본 AI 챗봇 서비스는 무엇인가요? 모두 선택해주세요.",
    "q25": "사용해 본 AI 챗봇 서비스 중 주로 사용하는 것은 무엇인가요?",
    "q26": "AI 챗봇 서비스를 주로 어떤 용도로 활용하셨거나, 앞으로 활용하고 싶으신가요?",
    "q27": "다음 두 서비스 중, 어느 서비스에 더 호감이 가나요? 현재 사용 여부는 고려하지 않고 응답해 주세요.",
    "q28": "현재 본인의 피부 상태에 얼마나 만족하시나요?",
    "q29": "한 달 기준으로 스킨케어 제품에 평균적으로 얼마나 소비하시나요?",
    "q30": "스킨케어 제품을 구매할 때 가장 중요하게 고려하는 요소는 무엇인가요?",
    "q31": "가장 스트레스를 많이 느끼는 상황은 무엇인가요?",
    "q32": "스트레스를 해소하는 방법으로 주로 사용하는 것은 무엇인가요?",
    "q33": "여러분은 요즘 가장 많이 사용하는 앱은 무엇인가요?",
    "q34": "여러분은 본인을 위해 소비하는 것 중 가장 기분 좋아지는 소비는 무엇인가요?",
    "q35": "여러분은 이사할 때 가장 스트레스 받는 부분은 어떤걸까요?",
    "q36": "여러분은 반려동물을 키우는 중이시거나 혹은 키워보신 적이 있으신가요?",
    "q37": "초등학생 시절 겨울방학 때 가장 기억에 남는 일은 무엇인가요?",
    "q38": "여러분이 가장 선호하는 설 선물 유형은 무엇인가요?",
    "q39": "여러분은 전통시장을 얼마나 자주 방문하시나요?",
    "q40": "여러분이 현재 이용 중인 OTT 서비스는 몇 개인가요?",
    "q41": "여러분은 평소 체력 관리를 위해 어떤 활동을 하고 계신가요? 모두 선택해주세요.",
}
# 역매핑: 한글문항문 -> q필드명
QUESTION_REVERSE = {v: k for k, v in QUESTION_MAP.items()}

# ──────────────────────────────────────────────────────────────────────────────
def none_if_empty(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return None if s == "" else s

def to_int_or_none(v):
    v = none_if_empty(v)
    if v is None:
        return None
    try:
        return int(float(v))
    except Exception:
        return None

def pick_question_field(header: str):
    """
    CSV 헤더(문항문)가 QUESTION_MAP에 있으면 q필드명(q1~q41) 반환.
    아니라면 None.
    """
    return QUESTION_REVERSE.get(header)

# ──────────────────────────────────────────────────────────────────────────────
class Command(BaseCommand):
    help = "CSV를 PanelRecord 테이블에 적재합니다 (메타 + q1~q41 매핑)."

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True, help="CSV 파일 경로")
        parser.add_argument("--sep", type=str, default=",", help="구분자(기본 ,)")
        parser.add_argument("--encoding", type=str, default="utf-8-sig", help="인코딩(기본 utf-8-sig)")
        parser.add_argument("--chunksize", type=int, default=5000, help="청크 크기(기본 5000)")
        parser.add_argument("--upsert", action="store_true", help="기존 PK 충돌 시 update_or_create(느릴 수 있음)")

    def handle(self, *args, **opts):
        path = opts["path"]
        sep = opts["sep"]
        enc = opts["encoding"]
        chunksize = opts["chunksize"]
        do_upsert = opts["upsert"]

        if not os.path.exists(path):
            raise CommandError(f"CSV 파일을 찾을 수 없습니다: {path}")

        # 첫 청크에서 컬럼 구조 파악
        first_iter = True
        total = 0
        created_total = 0

        try:
            reader = pd.read_csv(path, sep=sep, encoding=enc, dtype=str, chunksize=chunksize, keep_default_na=False)
        except Exception as e:
            raise CommandError(f"CSV 읽기 오류: {e}")

        for df in reader:
            # 문자열로 읽고 후처리(빈값→"")
            df = df.fillna("")

            if first_iter:
                cols = list(df.columns)
                # 문항 컬럼 후보: 헤더가 QUESTION_REVERSE에 있는 것들
                question_headers = [c for c in cols if c in QUESTION_REVERSE]
                missing_meta = [m for m in META_FIELDS if m not in cols]
                self.stdout.write(self.style.SUCCESS(f"총 컬럼 수: {len(cols)}"))
                self.stdout.write(f"메타 컬럼 누락: {missing_meta if missing_meta else '없음'}")
                self.stdout.write(f"문항 컬럼 수: {len(question_headers)} / 기대 41")
                if len(question_headers) < 41:
                    self.stdout.write(self.style.WARNING("⚠️ CSV에 없는 문항도 있습니다. 존재하는 문항만 적재합니다."))
                first_iter = False

            objs = []
            # 트랜잭션 처리
            with transaction.atomic():
                for _, row in df.iterrows():
                    # 필수: id
                    rid = none_if_empty(row.get("id"))
                    if not rid:
                        continue

                    # 메타 필드 매핑/캐스팅
                    gender = none_if_empty(row.get("gender"))
                    birth = to_int_or_none(row.get("birth"))
                    region = none_if_empty(row.get("region"))
                    subregion = none_if_empty(row.get("subregion"))
                    married = none_if_empty(row.get("married"))
                    nchild = to_int_or_none(row.get("nchild"))
                    famsize = none_if_empty(row.get("famsize"))
                    education_level = none_if_empty(row.get("education_level"))
                    job = none_if_empty(row.get("job"))
                    work = none_if_empty(row.get("work"))
                    p_income = none_if_empty(row.get("p_income"))
                    h_income = none_if_empty(row.get("h_income"))
                    owned_products = none_if_empty(row.get("owned_products"))
                    phone_brand = none_if_empty(row.get("phone_brand"))
                    phone_model = none_if_empty(row.get("phone_model"))
                    car_ownship = none_if_empty(row.get("car_ownship"))
                    car_manufacturer = none_if_empty(row.get("car_manufacturer"))
                    car_model = none_if_empty(row.get("car_model"))
                    ever_smoked = none_if_empty(row.get("ever_smoked"))
                    brand_smoked = none_if_empty(row.get("brand_smoked"))
                    brand_smoked_ETC = none_if_empty(row.get("brand_smoked_ETC"))
                    ever_esmoked = none_if_empty(row.get("ever_esmoked"))
                    ever_alcohol = none_if_empty(row.get("ever_alcohol"))
                    ever_smoked_brand_ETC = none_if_empty(row.get("ever_smoked_brand_ETC"))
                    ever_smoked_ETC = none_if_empty(row.get("ever_smoked_ETC"))
                    p_company = none_if_empty(row.get("p_company"))
                    loyalty = to_int_or_none(row.get("loyalty"))

                    # q1~q41 채우기: CSV 헤더가 문항문인 컬럼만 반영
                    q_values = {f"q{i}": None for i in range(1, 42)}
                    for header in QUESTION_REVERSE.keys():
                        if header in row:
                            q_field = pick_question_field(header)  # q1~q41
                            if q_field:
                                q_values[q_field] = none_if_empty(row.get(header))

                    if do_upsert:
                        # 느리지만 안전한 업서트
                        PanelRecord.objects.update_or_create(
                            id=rid,
                            defaults=dict(
                                gender=gender, birth=birth, region=region, subregion=subregion,
                                married=married, nchild=nchild, famsize=famsize,
                                education_level=education_level, job=job, work=work,
                                p_income=p_income, h_income=h_income,
                                owned_products=owned_products, phone_brand=phone_brand, phone_model=phone_model,
                                car_ownship=car_ownship, car_manufacturer=car_manufacturer, car_model=car_model,
                                ever_smoked=ever_smoked, brand_smoked=brand_smoked, brand_smoked_ETC=brand_smoked_ETC,
                                ever_esmoked=ever_esmoked, ever_alcohol=ever_alcohol,
                                ever_smoked_brand_ETC=ever_smoked_brand_ETC, ever_smoked_ETC=ever_smoked_ETC,
                                p_company=p_company,
                                loyalty=loyalty,
                                **q_values,
                            )
                        )
                        created_total += 1
                    else:
                        # bulk_create 용 객체 생성
                        objs.append(PanelRecord(
                            id=rid,
                            gender=gender, birth=birth, region=region, subregion=subregion,
                            married=married, nchild=nchild, famsize=famsize,
                            education_level=education_level, job=job, work=work,
                            p_income=p_income, h_income=h_income,
                            owned_products=owned_products, phone_brand=phone_brand, phone_model=phone_model,
                            car_ownship=car_ownship, car_manufacturer=car_manufacturer, car_model=car_model,
                            ever_smoked=ever_smoked, brand_smoked=brand_smoked, brand_smoked_ETC=brand_smoked_ETC,
                            ever_esmoked=ever_esmoked, ever_alcohol=ever_alcohol,
                            ever_smoked_brand_ETC=ever_smoked_brand_ETC, ever_smoked_ETC=ever_smoked_ETC,
                            p_company=p_company,
                            loyalty=loyalty,
                            **q_values,
                        ))
                        created_total += 1

                if objs:
                    # PK 충돌 무시하고 신규만 삽입
                    PanelRecord.objects.bulk_create(
                        objs, batch_size=1000, ignore_conflicts=True
                    )

            total += len(df)
            self.stdout.write(self.style.NOTICE(f"Processed rows: {total}"))

        self.stdout.write(self.style.SUCCESS(f"✅ 적재 완료: {created_total}행 처리 (파일: {path})"))
