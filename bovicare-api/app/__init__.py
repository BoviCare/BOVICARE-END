import asyncio
import os
import sys
from flask import Flask
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_path = os.path.join(project_root, "RAG")
if os.path.isdir(rag_path) and rag_path not in sys.path:
    sys.path.insert(0, rag_path)
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from os import environ

app = Flask(__name__)
app.config.from_object('config.Config')
db = SQLAlchemy(app)

try:
    from RAG.rag_service import RAGService
    rag_service = RAGService()
except Exception as rag_import_error:
    rag_service = None
    print(f"⚠️ Não foi possível carregar o RAGService: {rag_import_error}")

# Configuração CORS manual mais simples
@app.after_request
def after_request(response):
    # Allow the frontend origin and needed headers including our custom X-User-* headers
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-User-Id, X-User-Name')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

from app import routes
from app import api_v1

with app.app_context():
    db.create_all()
    print("✅ Banco de dados inicializado com sucesso!")

    should_start_rag = os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug

    # Tentativa segura de migrar a coluna profile_photo_url se não existir
    try:
        from sqlalchemy import text
        db.session.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='users' AND column_name='profile_photo_url'
                ) THEN
                    BEGIN
                        ALTER TABLE users ADD COLUMN profile_photo_url VARCHAR(500);
                    EXCEPTION WHEN OTHERS THEN
                        -- Ignorar erro caso a coluna já exista em alguns bancos
                        NULL;
                    END;
                END IF;
            END$$;
        """))
        db.session.commit()
    except Exception as mig_err:
        # Em SQLite, a sintaxe acima não vale; seguimos sem travar
        db.session.rollback()
        try:
            # Tentativa best-effort para SQLite: criar tabela temporária está fora de escopo; ignorar
            pass
        except Exception:
            pass

    if rag_service and should_start_rag:
        try:
            asyncio.run(rag_service.startup())
            print("✅ Serviço RAG inicializado")
        except RuntimeError:
            # já existe loop rodando; inicia de forma assíncrona
            loop = asyncio.get_event_loop()
            loop.create_task(rag_service.startup())
        except Exception as e:
            print(f"⚠️ Falha ao iniciar o serviço RAG: {e}")
