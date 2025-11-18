from django.db import models
from pgvector.django import VectorField

class InsightDocVec(models.Model):
    doc_id = models.TextField(null=True)
    chunk_index = models.IntegerField(null=True)
    content = models.TextField(null=True)
    embedding = VectorField(dimensions=1024, null=True)  # 차원수만 KURE에 맞게

    class Meta:
        db_table = "insight_docvec"
        managed = True       # ❗ 꼭 True
