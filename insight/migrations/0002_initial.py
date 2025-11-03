# insight/migrations/0002_initial.py
from django.db import migrations, models
import pgvector.django.vector

class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('insight', '0001_enable_pgvector'),
    ]

    operations = [
        migrations.CreateModel(
            name='ItemEmbedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.CharField(max_length=64, db_index=True)),      # unique 제거
                ('main', models.TextField(blank=True, null=True)),
                ('sub', models.TextField(blank=True, null=True)),
                ('qids_used', models.TextField(blank=True, null=True)),       # 추가
                ('vec', pgvector.django.vector.VectorField(dimensions=1024)), # 1024로
            ],
            options={
                'db_table': 'kure_item_embeddings_v2',
                'indexes': [
                    models.Index(fields=['uid', 'sub'], name='idx_uid_sub'), # 복합 인덱스
                ],
            },
        ),
    ]
