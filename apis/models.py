from django.db import models
from django.db.models import JSONField

class CustomerData(models.Model):
    user_id = models.CharField(max_length=50, unique=True)
    
    full_data = JSONField() 

    def __str__(self):
        return self.user_id