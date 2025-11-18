import csv
import json

from django.core.management.base import BaseCommand
from insight.models import InsightDocVec


class Command(BaseCommand):
    help = "chunks_embeddings_kure.csv 를 vecdb.insight_docvec 테이블에 로드"

    def add_arguments(self, parser):
        # CSV 경로
        parser.add_argument(
            "--path",
            type=str,
            default="chunks_embeddings_kure.csv",
            help="CSV 파일 경로 (manage.py 기준 상대 경로)",
        )
        # 배치 크기
        parser.add_argument(
            "--batch-size",
            type=int,
            default=500,
            help="bulk_create 배치 크기 (기본 500)",
        )
        # 사용할 DB alias
        parser.add_argument(
            "--database",
            type=str,
            default="vecdb",
            help="Django DB alias (기본값: vecdb)",
        )

    def handle(self, *args, **options):
        csv_path = options["path"]
        batch_size = options["batch_size"]
        db_alias = options["database"]

        self.stdout.write(self.style.WARNING(
            f"[{db_alias}] {csv_path} 로드 시작"
        ))

        # 기존 데이터 삭제
        InsightDocVec.objects.using(db_alias).all().delete()
        self.stdout.write(self.style.SUCCESS("기존 insight_docvec 데이터 삭제 완료"))

        objs = []
        total = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # ⚠️ CSV 헤더 이름에 맞게 수정 필요
                doc_id = row["doc_id"]
                chunk_index = int(row["chunk_index"])
                content = row["content"]

                # "[0.1, 0.2, ...]" 형태라고 가정
                emb_list = json.loads(row["embedding"])

                obj = InsightDocVec(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=content,
                    embedding=emb_list,
                )
                objs.append(obj)

                if len(objs) >= batch_size:
                    InsightDocVec.objects.using(db_alias).bulk_create(objs)
                    total += len(objs)
                    self.stdout.write(f"{total}개 적재 완료...")
                    objs = []

        # 남은 것들 처리
        if objs:
            InsightDocVec.objects.using(db_alias).bulk_create(objs)
            total += len(objs)

        self.stdout.write(self.style.SUCCESS(f"총 {total}개 로드 완료"))
