from django.core.management.base import BaseCommand, CommandError
from apis.models import CustomerData
from django.db import transaction
import pandas as pd

FIELD_MAP = {
    # '필드명': '데이터 타입' (int 또는 str)
    'gender': 'str', 'birth': 'int', 'region': 'str', 'subregion': 'str',
    'married': 'str', 'nchild': 'int', 'famsize': 'str', 'education_level': 'str',
    'job': 'str', 'work': 'str', 'p_income': 'str', 'h_income': 'str',
    'owned_products': 'str', 'phone_brand': 'str', 'phone_model': 'str',
    'car_ownship': 'str', 'car_manufacturer': 'str', 'car_model': 'str',
    'ever_smoked': 'str', 'brand_smoked': 'str', 'brand_smoked_ETC': 'str',
    'ever_esmoked': 'str', 'ever_smoked_brand_ETC': 'str', 'ever_alcohol': 'str',
    'p_company':'str', 'loyalty': 'int',
}
ID_COLUMN = 'id'

class Command(BaseCommand):
    help = 'Loads customer metadata from a CSV file into the CustomerData model JSONField.'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The path to the CSV file to load.')
        parser.add_argument(
            '--encoding', type=str, default='utf-8',
            help='The encoding of the CSV file (default: utf-8).'
        )

    def safe_int(self, value):
        if pd.notna(value):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                self.stdout.write(self.style.WARNING(f"경고: 숫자 변환 실패 (값: {value}). None으로 처리합니다."))
                return None
        return None

    def safe_str(self, value):
        if pd.notna(value):
            s = str(value).strip()
            return s if s != '' else None
        return None

    def handle(self, *args, **options):
        file_path = options['csv_file']
        encoding = options['encoding']

        self.stdout.write(f"데이터 파일 로드를 시작합니다: {file_path} (encoding={encoding})")

        # 파일 로드
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except FileNotFoundError:
            raise CommandError(f'파일을 찾을 수 없습니다: "{file_path}"')
        except Exception as e:
            raise CommandError(f'CSV 파일 로드 중 오류 발생 (encoding={encoding}): {e}')

        # 컬럼 이름 공백 제거
        df.columns = df.columns.str.strip()

        # 2) 저장 로직
        records_created = 0
        records_updated = 0
        records_skipped_null_id = 0
        records_failed = 0

        for index, row in df.iterrows():
            # 주 식별자
            user_id_raw = row.get(ID_COLUMN)
            if pd.isna(user_id_raw) or str(user_id_raw).strip() == '':
                records_skipped_null_id += 1
                continue
            user_id = str(user_id_raw).strip()

            # 메타데이터 값이 있지 않으면 JSON에 '키 자체'를 넣지 않음
            metadata = {}
            for key, data_type in FIELD_MAP.items():
                value = row.get(key)
                processed = self.safe_int(value) if data_type == 'int' else self.safe_str(value)
                if processed is not None:          # ← 중요: None이면 아예 건너뜀(키 미생성)
                    metadata[key] = processed

            # 행 단위 트랜잭션
            try:
                with transaction.atomic():
                    obj, created = CustomerData.objects.update_or_create(
                        user_id=user_id,
                        defaults={'full_data': metadata}
                    )
                    if created: records_created += 1
                    else:       records_updated += 1
            except Exception as e:
                records_failed += 1
                self.stdout.write(self.style.ERROR(
                    f"Index {index} (user_id={user_id}) 처리 중 오류: {e} - 행 스킵"
                ))

        # 결과 출력
        self.stdout.write("-" * 50)
        self.stdout.write(self.style.SUCCESS("✅ 데이터 저장 완료!"))
        self.stdout.write(f"총 CSV 행: {len(df)}")
        self.stdout.write(f"생성된 DB 레코드(신규): {records_created}개")
        self.stdout.write(f"업데이트된 DB 레코드: {records_updated}개")
        self.stdout.write(f"ID가 없어 건너뛴 행: {records_skipped_null_id}개")
        self.stdout.write(f"에러로 스킵된 행: {records_failed}개")
        self.stdout.write("-" * 50)
