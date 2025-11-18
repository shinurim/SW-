class VectorDBRouter:
    # 벡터 DB를 사용하는 모델 이름을 명시적으로 지정
    vector_models = ['itemembedding', 'docvec']
    
    def db_for_read(self, model, **hints):
        if model._meta.model_name in self.vector_models:
            return 'vecdb'
        # 벡터 모델이 아니면 반드시 'default'로 지정
        return 'default'

    def db_for_write(self, model, **hints):
        if model._meta.model_name in self.vector_models:
            return 'vecdb'
        # 벡터 모델이 아니면 반드시 'default'로 지정
        return 'default'

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if model_name in self.vector_models:
            # 벡터 모델은 'vecdb'에서만 마이그레이션 허용
            return db == 'vecdb'
        
        # 나머지 모든 모델은 'default' DB에서만 마이그레이션 허용
        # (중요: 다른 DB에서 마이그레이션되는 것을 명시적으로 차단)
        return db == 'default'