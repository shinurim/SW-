from django.db import models
from django.db.models import JSONField

class CustomerData(models.Model):
    user_id = models.CharField(max_length=50, unique=True)
    full_data = JSONField() 

    def __str__(self):
        return self.user_id

class UserData(models.Model):
    uuid = models.IntegerField(null=True, blank=True)
    user_id = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    name =  models.CharField(max_length=100, blank=True, null=True)
    email =  models.CharField(max_length=200, blank=True, null=True)
    phone_number = models.CharField(max_length=200, blank=True, null=True)
    class Meta:
        db_table = "users"
    

class SegmentHistory(models.Model):
    # 로그인 붙이면 ForeignKey로 바꿔도 됨
    user_id = models.CharField(max_length=100, blank=True, null=True)

    # 프론트에서 segment 이름/title 받아서 저장
    segment_name = models.CharField(max_length=255)

    # 질의 요약용
    user_input = models.TextField(blank=True, null=True)
    main = models.CharField(max_length=50, blank=True, null=True)
    sub  = models.CharField(max_length=50, blank=True, null=True)

    # stage3 전체 스냅샷
    stage3 = models.JSONField()
    # insight 전체 스냅샷 (지금 generate_insight에서 만든 payload)
    insight = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "segment_history"

    def __str__(self):
        return f"{self.segment_name} ({self.id})"