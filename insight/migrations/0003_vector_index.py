# insight/migrations/0003_vector_index.py
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('insight', '0002_initial'),
    ]
    operations = [
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS item_vec_hnsw "
            "ON kure_item_embeddings_v2 USING hnsw (vec vector_cosine_ops) "
            "WITH (m=16, ef_construction=64);"
        ),
    ]
