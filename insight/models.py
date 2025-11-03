from django.db import models
from pgvector.django import VectorField

EMB_DIM1 = 768

class ItemEmbedding(models.Model):
    uid  = models.CharField(max_length=64, db_index=True)
    main = models.TextField(null=True, blank=True)
    sub  = models.TextField(null=True, blank=True)
    qids_used = models.TextField(null=True, blank=True)
    vec  = VectorField(dimensions=EMB_DIM1)

    class Meta:
        db_table = "kure_item_embeddings_v2"
        indexes = [
            models.Index(fields=["uid", "sub"]),  # ✅ uid + sub 복합 인덱스
        ]

    def __str__(self):
        return f"{self.uid} | {self.sub}"

    
EMB_DIM2 = 1536  

class DocVec(models.Model):
    content = models.TextField()
    vec = VectorField(dimensions=EMB_DIM2)