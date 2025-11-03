from django.core.management.base import BaseCommand
from django.db import transaction
from langchain_chroma import Chroma
from insight.models import DocVec  # ✅ models 위치 정확히 맞게

class Command(BaseCommand):
    help = "Chroma → vecdb 적재"

    def add_arguments(self, parser):
        parser.add_argument("--persist_dir", required=True, help="Chroma persist_directory")
        parser.add_argument("--truncate", action="store_true")

    def handle(self, *args, **opts):
        persist_dir = opts["persist_dir"]
        truncate = opts["truncate"]

        db = Chroma(persist_directory=persist_dir, embedding_function=None)
        data = db.get(include=["embeddings", "documents"])

        embeddings = data.get("embeddings", [])
        documents = data.get("documents", [])

        if truncate:
            DocVec.objects.using("vecdb").all().delete()

        rows = []
        for doc, emb in zip(documents, embeddings):
            if emb is None or not doc or not doc.strip():
                continue
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            rows.append(DocVec(content=doc, vec=emb))

        with transaction.atomic(using="vecdb"):
            DocVec.objects.using("vecdb").bulk_create(rows, batch_size=1000)

        self.stdout.write(self.style.SUCCESS(f"✅ vecdb에 {len(rows)}개 Chroma 벡터 적재 완료"))
