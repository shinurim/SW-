# insight/migrations/0004_update_itemembedding.py
from django.db import migrations, models
import pgvector.django.vector

class Migration(migrations.Migration):
    dependencies = [
        ('insight', '0003_vector_index'),
    ]

    operations = [
        # 1) uid에서 unique 제거 (고유 제약 해제)
        migrations.AlterField(
            model_name='itemembedding',
            name='uid',
            field=models.CharField(max_length=64, db_index=True),
        ),

        # 2) qids_used 필드 추가 (어떤 q번호들로 평균 냈는지 기록)
        migrations.AddField(
            model_name='itemembedding',
            name='qids_used',
            field=models.TextField(blank=True, null=True),
        ),

        # 3) (uid, sub) 복합 인덱스 추가
        migrations.AddIndex(
            model_name='itemembedding',
            index=models.Index(fields=['uid', 'sub'], name='idx_uid_sub'),
        ),

        # (선택) 4) vec 차원 변경(모델과 일치) — DB 타입엔 실변경 없음, Django 레벨 메타만 갱신
        migrations.AlterField(
            model_name='itemembedding',
            name='vec',
            field=pgvector.django.vector.VectorField(dimensions=1024),
        ),
    ]
