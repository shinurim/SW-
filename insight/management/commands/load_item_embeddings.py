# insight/management/commands/load_item_embeddings.py
import pandas as pd
import ast
from django.core.management.base import BaseCommand, CommandError
from insight.models import ItemEmbedding 
from tqdm import tqdm # 💡 tqdm 추가

class Command(BaseCommand):
    help = 'Loads ItemEmbedding data from a specified CSV file path.'

    def add_arguments(self, parser):
        # CSV 파일 경로를 필수 인자로 받도록 명시적으로 정의합니다.
        parser.add_argument('csv_path', type=str, help='Path to the CSV file containing embeddings.')
        parser.add_argument('--truncate', action='store_true', help='Delete all existing ItemEmbedding records before loading.')


    def handle(self, *args, **options):
        csv_path = options['csv_path']
        truncate = options['truncate']
        
        self.stdout.write(f"Reading data from: {csv_path}")

        # --- 1. CSV 파일 로드 (인코딩 및 엔진 오류 처리) ---
        try:
            data_df = pd.read_csv(
                csv_path, 
                encoding='utf-8', 
                engine='python' 
            )
        except UnicodeDecodeError:
            self.stdout.write("UTF-8 decoding failed. Trying CP949...")
            data_df = pd.read_csv(
                csv_path, 
                encoding='cp949', 
                engine='python'
            )
        except FileNotFoundError:
            raise CommandError(f"CSV file not found at {csv_path}")
        except Exception as e:
            raise CommandError(f"CSV 로드 실패: {e}. Try checking file structure.")

        total_rows = len(data_df)
        self.stdout.write(f"Successfully loaded {total_rows} rows into DataFrame.")
        
        # --- 2. 기존 데이터 처리 ---
        if truncate:
            self.stdout.write(self.style.WARNING("Truncate option enabled. Deleting all existing ItemEmbedding records..."))
            ItemEmbedding.objects.using('vecdb').all().delete()
            self.stdout.write(self.style.SUCCESS("Deletion complete."))


        # --- 3. 객체 생성 및 Bulk Insert (tqdm 적용) ---
        objects_to_create = []
        
        # 💡 tqdm을 사용하여 진행 막대 표시
        for index, row in tqdm(data_df.iterrows(), total=total_rows, desc="Processing and Creating Objects"):
            try:
                # 'sub_vec' 문자열을 float 리스트로 안전하게 변환 (가장 시간이 오래 걸리는 작업)
                vector_list = ast.literal_eval(row['sub_vec']) 
            except (ValueError, SyntaxError, KeyError) as e:
                self.stderr.write(self.style.ERROR(f"Error parsing vector for row {index}: {e} - Skipping."))
                continue

            objects_to_create.append(
                ItemEmbedding(
                    uid=row['uid'],
                    main=row['main'],
                    sub=row['sub'],
                    qids_used=row.get('qids_used'), 
                    vec=vector_list
                )
            )

        self.stdout.write(f"Vector parsing and object creation finished. Starting database bulk insert...")

        # 💡💡 메모리 오류 해결을 위해 객체를 5000개씩 나눠서 삽입합니다. 💡💡
        BATCH_SIZE = 250
        total_objects = len(objects_to_create)
        
        # tqdm을 사용하여 Bulk Insert 진행 상황 표시
        # range(start, stop, step)을 사용하여 인덱스를 5000 단위로 건너뜁니다.
        for i in tqdm(range(0, total_objects, BATCH_SIZE), desc="Database Bulk Inserting"):
            # 현재 배치(5000개) 객체를 슬라이싱
            batch = objects_to_create[i:i + BATCH_SIZE]
            
            # bulk_create를 사용하여 vecdb 연결에 삽입
            # 매 반복마다 작은 트랜잭션이 생성되어 메모리 부하를 줄입니다.
            ItemEmbedding.objects.using('vecdb').bulk_create(
                batch, 
                ignore_conflicts=True, 
                batch_size=BATCH_SIZE 
            )

        self.stdout.write(self.style.SUCCESS(
            f"Successfully loaded {total_objects} ItemEmbedding vectors into 'vecdb'."
        ))