import asyncio
from flask import request, jsonify, make_response
from app import app, db
from app.models import User, PasswordReset, Animal, Weighing, Activity, Herd, UserHerd
from app import rag_service
# ===== ROTA PARA CHAT IA (RAG) =====

@app.route('/api/chat/diagnose', methods=['POST'])
def chat_diagnose():
    try:
        if request.method == 'OPTIONS':
            return make_response()

        data = request.get_json() or {}
        message = data.get('message') or data.get('query')

        if not message or not isinstance(message, str) or not message.strip():
            return make_response(jsonify({'message': 'Mensagem inválida'}), 400)

        if rag_service is None:
            return make_response(jsonify({'message': 'Serviço de diagnóstico indisponível'}), 503)

        result = asyncio.run(rag_service.ask(message.strip(), top_k=5))

        return make_response(jsonify({
            'reply': result.get('answer', ''),
            'sources': result.get('sources', [])
        }), 200)

    except Exception as e:
        print(f"DEBUG: Erro no chat diagnose: {str(e)}")
        return make_response(jsonify({'message': f'Erro ao processar diagnóstico: {str(e)}'}), 500)

from app.email_service import email_service, sms_service
import os
from werkzeug.utils import secure_filename
from datetime import datetime

# Decorator para CORS
def cors_headers(f):
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            response = make_response()
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            return response
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/', methods=['GET'])
def home():
    return make_response(jsonify({
        'message': 'BoviCare API is running!', 
        'status': 'online',
        'endpoints': {
            'test': '/test',
            'test_cors': '/test-cors',
            'register': '/users/register',
            'login': '/users/login',
            'users': '/users',
            'user_by_id': '/users/<id>',
            'api_v1': '/api/v1/',
            'dashboard': '/api/v1/dashboard',
            'herds': '/api/v1/herds',
            'animals': '/api/v1/animals',
            'weighings': '/api/v1/animals/<id>/weighings',
            'movements': '/api/v1/animals/<id>/movements',
            'reproductions': '/api/v1/animals/<id>/reproductions',
            'vaccines': '/api/v1/vaccines',
            'health': '/api/v1/animals/<id>/health',
            'attachments': '/api/v1/animals/<id>/attachments'
        },
        'methods': {
            'register': 'POST',
            'login': 'POST',
            'users': 'GET',
            'user_by_id': 'GET, PUT, DELETE',
            'api_v1': 'GET, POST, PUT, DELETE'
        }
    }), 200)

@app.route('/test', methods=['GET'])
def test():
    return make_response(jsonify({'message': 'test route'}), 200)

@app.route('/test-cors', methods=['GET', 'POST', 'OPTIONS'])
def test_cors():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response
    
    return make_response(jsonify({
        'message': 'CORS test successful',
        'method': request.method,
        'headers': dict(request.headers)
    }), 200)

@app.route('/users/register', methods=['POST'])
def create_user():
    try:
        print(f"DEBUG: Recebendo requisição POST em /users/register")
        print(f"DEBUG: Headers: {dict(request.headers)}")
        
        data = request.get_json()
        print(f"DEBUG: Dados recebidos: {data}")
        
        if not data.get('username') or not data.get('email') or not data.get('password'):
            print(f"DEBUG: Dados faltando - username: {data.get('username')}, email: {data.get('email')}, password: {bool(data.get('password'))}")
            return make_response(jsonify({'message': 'Missing data'}), 400)

        # Verificar se usuário já existe
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user:
            return make_response(jsonify({'message': 'Email já cadastrado'}), 400)

        # Hash seguro da senha antes de salvar
        from werkzeug.security import generate_password_hash
        hashed_password = generate_password_hash(data['password'])

        new_user = User(
            username=data['username'],
            email=data['email'],
            password=hashed_password,
            role=data.get('role', 'user'),
            phone=data.get('phone')
        )

        db.session.add(new_user)
        db.session.commit()
        
        print(f"DEBUG: Usuário criado com sucesso: {new_user.id}")

        return make_response(jsonify({
            'message': 'User created successfully',
            'user': new_user.json()
        }), 201)
    except Exception as e:
        print(f"DEBUG: Erro ao criar usuário: {str(e)}")
        return make_response(jsonify({'message': f'Error creating user: {str(e)}'}), 500)

@app.route('/users/login', methods=['POST'])
def login_user():
    try:
        print(f"DEBUG: Recebendo requisição POST em /users/login")
        print(f"DEBUG: Headers: {dict(request.headers)}")
        
        data = request.get_json()
        print(f"DEBUG: Dados de login recebidos: {data}")
        
        if not data.get('email') or not data.get('password'):
            print(f"DEBUG: Dados faltando - email: {data.get('email')}, password: {bool(data.get('password'))}")
            return make_response(jsonify({'message': 'Missing email or password'}), 400)

        # Buscar usuário por email
        user = User.query.filter_by(email=data['email']).first()
        print(f"DEBUG: Usuário encontrado: {user is not None}")
        
        if user:
            from werkzeug.security import check_password_hash

            is_hash_valid = False
            try:
                is_hash_valid = check_password_hash(user.password, data['password'])
            except Exception as e:
                print(f"DEBUG: check_password_hash falhou: {str(e)}")
                is_hash_valid = False

            if is_hash_valid:
                print(f"DEBUG: Login bem-sucedido para usuário: {user.id} (hash)")
                return make_response(jsonify({
                    'message': 'Login successful',
                    'user': user.json(),
                    'success': True,
                    # token fictício para front atual
                    'token': 'dev-token'
                }), 200)

        
        print(f"DEBUG: Login falhou - usuário não encontrado ou senha incorreta")
        return make_response(jsonify({
            'message': 'Invalid email or password',
            'success': False
        }), 401)

    except Exception as e:
        print(f"DEBUG: Erro ao fazer login: {str(e)}")
        return make_response(jsonify({
            'message': f'Error logging in: {str(e)}',
            'success': False
        }), 500)

@app.route('/users', methods=['GET'])
def get_users():
    try:
        print('DEBUG: Received GET /users request from', request.remote_addr)
        users = User.query.all()
        return make_response(jsonify([user.json() for user in users]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Error fetching users: {str(e)}'}), 500)

@app.route('/users/<int:id>', methods=['GET'])
def get_user(id):
    try:
        user = User.query.filter_by(id=id).first()
        if user:
            return make_response(jsonify({'user': user.json()}), 200)
        return make_response(jsonify({'message': 'User not found'}), 404)
    except Exception as e:
        return make_response(jsonify({'message': f'Error fetching user: {str(e)}'}), 500)

@app.route('/users/<int:id>', methods=['PUT'])
def update_user(id):
    try:
        user = User.query.filter_by(id=id).first()
        if user:
            data = request.get_json()
            user.username = data.get('username', user.username)
            user.email = data.get('email', user.email)
            user.phone = data.get('phone', user.phone)  # Adicionar telefone
            user.role = data.get('role', user.role)  # Permitir atualizar role
            db.session.commit()
            return make_response(jsonify({'message': 'User updated'}), 200)
        return make_response(jsonify({'message': 'User not found'}), 404)
    except Exception as e:
        return make_response(jsonify({'message': f'Error updating user: {str(e)}'}), 500)

@app.route('/users/<int:id>', methods=['DELETE'])
def delete_user(id):
    try:
        user = User.query.filter_by(id=id).first()
        if user:
            db.session.delete(user)
            db.session.commit()
            return make_response(jsonify({'message': 'User deleted'}), 200)
        return make_response(jsonify({'message': 'User not found'}), 404)
    except Exception as e:
        return make_response(jsonify({'message': f'Error deleting user: {str(e)}'}), 500)

# ===== ROTAS DE RECUPERAÇÃO DE SENHA =====

@app.route('/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Inicia processo de recuperação de senha"""
    try:
        print(f"DEBUG: Recebendo requisição POST em /auth/forgot-password")
        
        data = request.get_json()
        print(f"DEBUG: Dados recebidos: {data}")
        
        method = data.get('method')  # 'email' ou 'sms'
        email = data.get('email')
        phone = data.get('phone')
        
        if not method or (method not in ['email', 'sms']):
            return make_response(jsonify({'message': 'Método inválido'}), 400)
        
        # Buscar usuário
        user = None
        if method == 'email' and email:
            user = User.query.filter_by(email=email).first()
        elif method == 'sms' and phone:
            user = User.query.filter_by(phone=phone).first()
        
        if not user:
            return make_response(jsonify({'message': 'Usuário não encontrado'}), 404)
        
        # Invalidar códigos anteriores do usuário
        PasswordReset.query.filter_by(user_id=user.id, used=False).update({'used': True})
        
        # Criar novo código de recuperação
        reset_request = PasswordReset(user_id=user.id, method=method)
        db.session.add(reset_request)
        db.session.commit()
        
        # Enviar código
        success = False
        if method == 'email':
            success = email_service.send_password_reset_email(
                user.email, 
                reset_request.code, 
                user.username
            )
        elif method == 'sms':
            success = sms_service.send_password_reset_sms(
                user.phone, 
                reset_request.code, 
                user.username
            )
        
        if success:
            return make_response(jsonify({
                'message': 'Código enviado com sucesso',
                'method': method,
                'expires_in': 30  # minutos
            }), 200)
        else:
            return make_response(jsonify({'message': 'Erro ao enviar código'}), 500)
            
    except Exception as e:
        print(f"DEBUG: Erro em forgot-password: {str(e)}")
        return make_response(jsonify({'message': f'Erro interno: {str(e)}'}), 500)

@app.route('/auth/verify-code', methods=['POST'])
def verify_code():
    """Verifica código de recuperação"""
    try:
        print(f"DEBUG: Recebendo requisição POST em /auth/verify-code")
        
        data = request.get_json()
        code = data.get('code')
        method = data.get('method')
        email = data.get('email')
        phone = data.get('phone')
        
        if not code or not method:
            return make_response(jsonify({'message': 'Código e método são obrigatórios'}), 400)
        
        # Buscar usuário
        user = None
        if method == 'email' and email:
            user = User.query.filter_by(email=email).first()
        elif method == 'sms' and phone:
            user = User.query.filter_by(phone=phone).first()
        
        if not user:
            return make_response(jsonify({'message': 'Usuário não encontrado'}), 404)
        
        # Buscar código válido
        reset_request = PasswordReset.query.filter_by(
            user_id=user.id,
            code=code,
            method=method,
            used=False
        ).first()
        
        if not reset_request or not reset_request.is_valid():
            return make_response(jsonify({'message': 'Código inválido ou expirado'}), 400)
        
        # Marcar código como usado
        reset_request.used = True
        db.session.commit()
        
        return make_response(jsonify({
            'message': 'Código verificado com sucesso',
            'user_id': user.id,
            'token': f"reset_{user.id}_{reset_request.id}"  # Token temporário para redefinir senha
        }), 200)
        
    except Exception as e:
        print(f"DEBUG: Erro em verify-code: {str(e)}")
        return make_response(jsonify({'message': f'Erro interno: {str(e)}'}), 500)

@app.route('/auth/reset-password', methods=['POST'])
def reset_password():
    """Redefine a senha do usuário"""
    try:
        print(f"DEBUG: Recebendo requisição POST em /auth/reset-password")
        
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('new_password')
        
        if not token or not new_password:
            return make_response(jsonify({'message': 'Token e nova senha são obrigatórios'}), 400)
        
        # Validar token (formato: reset_userid_resetid)
        if not token.startswith('reset_'):
            return make_response(jsonify({'message': 'Token inválido'}), 400)
        
        try:
            parts = token.split('_')
            user_id = int(parts[1])
            reset_id = int(parts[2])
        except (IndexError, ValueError):
            return make_response(jsonify({'message': 'Token inválido'}), 400)
        
        # Buscar usuário e reset request
        user = User.query.filter_by(id=user_id).first()
        reset_request = PasswordReset.query.filter_by(id=reset_id, user_id=user_id).first()
        
        if not user or not reset_request or not reset_request.used:
            return make_response(jsonify({'message': 'Token inválido ou expirado'}), 400)
        
        # Atualizar senha com hash
        from werkzeug.security import generate_password_hash
        user.password = generate_password_hash(new_password)
        db.session.commit()
        
        print(f"DEBUG: Senha atualizada para usuário: {user.id}")
        
        return make_response(jsonify({
            'message': 'Senha redefinida com sucesso'
        }), 200)
        
    except Exception as e:
        print(f"DEBUG: Erro em reset-password: {str(e)}")
        return make_response(jsonify({'message': f'Erro interno: {str(e)}'}), 500)

# ===== ROTA PARA UPLOAD DE FOTO DE PERFIL =====

@app.route('/api/profile/photo', methods=['POST', 'OPTIONS'])
def upload_profile_photo():
    """Upload de foto de perfil"""
    try:
        # Tratar CORS - não adicionar headers aqui pois já são adicionados globalmente
        if request.method == 'OPTIONS':
            return make_response()
        
        print(f"DEBUG: Recebendo requisição POST em /api/profile/photo")
        print(f"DEBUG: Headers: {dict(request.headers)}")
        print(f"DEBUG: Files: {list(request.files.keys())}")
        
        # Verificar se há arquivo na requisição
        if 'photo' not in request.files:
            print(f"DEBUG: Arquivo 'photo' não encontrado")
            return make_response(jsonify({'message': 'Nenhum arquivo enviado'}), 400)
        
        file = request.files['photo']
        
        # Verificar se o arquivo foi selecionado
        if file.filename == '':
            return make_response(jsonify({'message': 'Nenhum arquivo selecionado'}), 400)
        
        # Verificar tipo de arquivo
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return make_response(jsonify({'message': 'Tipo de arquivo não permitido'}), 400)
        
        # Verificar tamanho do arquivo (máximo 5MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 5 * 1024 * 1024:  # 5MB
            return make_response(jsonify({'message': 'Arquivo muito grande (máximo 5MB)'}), 400)
        
        # Criar diretório de uploads se não existir
        upload_folder = os.path.join(os.getcwd(), 'uploads', 'profiles')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Gerar nome único para o arquivo
        filename = secure_filename(file.filename)
        unique_filename = f"profile_{int(datetime.now().timestamp())}_{filename}"
        file_path = os.path.join(upload_folder, unique_filename)
        
        # Salvar arquivo
        file.save(file_path)
        
        # URL relativa para acessar a imagem
        photo_url = f"/uploads/profiles/{unique_filename}"

        # Tentar identificar usuário atual pelo header
        user_id = request.headers.get('X-User-ID') or request.headers.get('X-User-Id') or request.args.get('user_id')
        if user_id and str(user_id).isdigit():
            try:
                user = User.query.filter_by(id=int(user_id)).first()
                if user:
                    user.profile_photo_url = photo_url
                    db.session.commit()
            except Exception as save_err:
                db.session.rollback()
                print(f"DEBUG: Falha ao salvar URL da foto no usuário {user_id}: {save_err}")

        return make_response(jsonify({
            'message': 'Foto de perfil atualizada com sucesso',
            'photo_url': photo_url
        }), 200)
        
    except Exception as e:
        print(f"DEBUG: Erro em upload de foto: {str(e)}")
        return make_response(jsonify({'message': f'Erro interno: {str(e)}'}), 500)

# ===== ROTA PARA SERVIR ARQUIVOS ESTÁTICOS =====

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Servir arquivos de upload"""
    from flask import send_from_directory
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    print(f"DEBUG: Servindo arquivo: {filename}")
    print(f"DEBUG: Pasta de upload: {upload_folder}")
    print(f"DEBUG: Arquivo existe: {os.path.exists(os.path.join(upload_folder, filename))}")
    return send_from_directory(upload_folder, filename)

# ===== ENDPOINTS ADICIONAIS PARA O FRONTEND =====

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Obter dados do perfil do usuário"""
    try:
        # Obter ID do usuário do header ou parâmetro
        header_user_id = request.headers.get('X-User-ID') or request.headers.get('X-User-Id')
        param_user_id = request.args.get('user_id')
        user_id = None
        if header_user_id and str(header_user_id).isdigit():
            user_id = int(header_user_id)
        elif param_user_id and str(param_user_id).isdigit():
            user_id = int(param_user_id)

        if user_id is None:
            return make_response(jsonify({'message': 'ID do usuário não fornecido'}), 400)
        
        # Buscar usuário no banco de dados
        user = User.query.filter_by(id=user_id).first()
        
        if not user:
            return make_response(jsonify({'message': 'Usuário não encontrado'}), 404)
        
        # Dados reais do perfil
        profile_data = {
            'fullName': user.username or 'Nome não informado',
            'email': user.email or 'Email não informado',
            'phone': user.phone or '',
            'location': getattr(user, 'location', 'Localização não informada'),
            'cpf': getattr(user, 'cpf', 'CPF não informado'),
            'birthDate': getattr(user, 'birth_date', 'Data de nascimento não informada'),
            'registrationDate': user.created_at.strftime('%d/%m/%Y') if user.created_at else 'Data não informada',
            'lastAccess': 'Último acesso não informado',
            'photo_url': getattr(user, 'profile_photo_url', None)
        }
        return make_response(jsonify(profile_data), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao obter perfil: {str(e)}'}), 500)

@app.route('/api/user/current', methods=['GET'])
def get_current_user():
    """Obter dados do usuário atual"""
    try:
        # Obter ID do usuário do header Authorization ou parâmetro
        user_id = request.headers.get('X-User-ID')
        
        if not user_id:
            # Se não tiver header, tentar obter do parâmetro da query
            user_id = request.args.get('user_id')
        
        if not user_id:
            return make_response(jsonify({'message': 'ID do usuário não fornecido'}), 400)
        
        # Buscar usuário no banco de dados
        user = User.query.filter_by(id=int(user_id)).first()
        
        if not user:
            return make_response(jsonify({'message': 'Usuário não encontrado'}), 404)
        
        # Buscar fazenda do usuário (se existir)
        farm_name = None
        herd = Herd.query.join(UserHerd).filter(UserHerd.user_id == user.id).first()
        if herd:
            farm_name = herd.name
        
        current_user = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'phone': user.phone or '',
            'role': user.role,
            'farm_name': farm_name or 'Fazenda não cadastrada',
            'location': getattr(user, 'location', 'Localização não informada'),
            'registration_date': user.created_at.strftime('%d/%m/%Y') if user.created_at else 'Data não informada',
            'profile_photo_url': getattr(user, 'profile_photo_url', None)
        }
        
        return make_response(jsonify(current_user), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao obter usuário atual: {str(e)}'}), 500)

@app.route('/api/user/stats', methods=['GET'])
def get_user_stats():
    """Obter estatísticas do usuário"""
    try:
        # Obter ID do usuário do header ou parâmetro
        user_id = request.headers.get('X-User-ID') or request.args.get('user_id')
        
        if not user_id:
            return make_response(jsonify({'message': 'ID do usuário não fornecido'}), 400)
        
        # Buscar usuário no banco de dados
        user = User.query.filter_by(id=int(user_id)).first()
        
        if not user:
            return make_response(jsonify({'message': 'Usuário não encontrado'}), 404)
        
        # Buscar estatísticas reais (por enquanto com dados básicos)
        from app.models import Herd
        
        farms_count = Herd.query.join(UserHerd).filter(UserHerd.user_id == user.id).count() if hasattr(user, 'id') else 0
        cattle_count = Animal.query.filter(Animal.user_id == user.id).count() if hasattr(user, 'id') else 0
        
        return make_response(jsonify({
            'farmsRegistered': farms_count,
            'cattleRegistered': cattle_count
        }), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao obter estatísticas: {str(e)}'}), 500)

@app.route('/api/user/profile', methods=['PUT', 'OPTIONS'])
def update_user_profile():
    """Atualizar perfil do usuário"""
    try:
        if request.method == 'OPTIONS':
            return make_response()
        
        print(f"DEBUG: Recebendo requisição PUT em /api/user/profile")
        print(f"DEBUG: Dados recebidos: {request.json}")
        
        # Dados que seriam atualizados no banco de dados
        updated_data = request.json
        
        # TODO: Salvar dados no banco de dados
        
        return make_response(jsonify({
            'message': 'Perfil atualizado com sucesso',
            'data': updated_data
        }), 200)
        
    except Exception as e:
        print(f"DEBUG: Erro ao atualizar perfil: {str(e)}")
        return make_response(jsonify({'message': f'Erro ao atualizar perfil: {str(e)}'}), 500)

@app.route('/api/user/change-password', methods=['PUT', 'OPTIONS'])
def change_password():
    """Alterar senha do usuário"""
    try:
        if request.method == 'OPTIONS':
            return make_response()
        
        print(f"DEBUG: Recebendo requisição PUT em /api/user/change-password")
        print(f"DEBUG: Dados recebidos: {request.json}")
        
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        user_id = data.get('user_id')
        
        if not current_password or not new_password or not user_id:
            return make_response(jsonify({'message': 'Senha atual, nova senha e ID do usuário são obrigatórios'}), 400)
        
        # Buscar usuário no banco de dados
        user = User.query.filter_by(id=int(user_id)).first()
        
        if not user:
            return make_response(jsonify({'message': 'Usuário não encontrado'}), 404)
        
        # Verificar senha atual
        from werkzeug.security import check_password_hash
        if not check_password_hash(user.password, current_password):
            return make_response(jsonify({'message': 'Senha atual incorreta'}), 400)
        
        # Atualizar senha com hash
        from werkzeug.security import generate_password_hash
        user.password = generate_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        print(f"DEBUG: Senha alterada para usuário: {user.id}")
        
        return make_response(jsonify({
            'message': 'Senha alterada com sucesso'
        }), 200)
        
    except Exception as e:
        print(f"DEBUG: Erro ao alterar senha: {str(e)}")
        return make_response(jsonify({'message': f'Erro interno: {str(e)}'}), 500)

# ===== ROTAS PARA CADASTRO DE GADO =====

@app.route('/api/cattle', methods=['POST'])
def add_cattle():
    """Cadastrar novo gado"""
    try:
        data = request.get_json()
        header_user_id = request.headers.get('X-User-Id') or request.headers.get('X-User-ID')
        payload_user_id = data.get('user_id')
        owner_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            owner_user_id = int(header_user_id)
        elif payload_user_id and str(payload_user_id).isdigit():
            owner_user_id = int(payload_user_id)

        if owner_user_id is None:
            return make_response(jsonify({'message': 'Usuário não informado para criação do gado'}), 400)
        
        # Validar campos obrigatórios
        required_fields = [
            'name', 'entryDate', 'origin', 'gender', 'breed', 
            'category', 'entryWeight', 'birthDate', 'targetWeight', 'estimatedSlaughter'
        ]
        
        for field in required_fields:
            if not data.get(field):
                return make_response(jsonify({
                    'message': f'Campo obrigatório não fornecido: {field}'
                }), 400)
        
        # Criar novo animal no banco de dados
        
        entry_weight_value = data.get('entryWeight', data.get('entry_weight'))
        target_weight_value = data.get('targetWeight', data.get('target_weight'))
        herd_id = data.get('herdId') or data.get('herd_id')

        if herd_id:
            herd = Herd.query.join(UserHerd).filter(UserHerd.user_id == owner_user_id, Herd.id == herd_id).first()
            if not herd:
                return make_response(jsonify({'message': 'Fazenda não encontrada para o usuário informado'}), 404)

        new_animal = Animal(
            earring=data['name'],  # Usar nome como brinco temporariamente
            name=data['name'],
            breed=data['breed'],
            birth_date=datetime.strptime(data['birthDate'], '%Y-%m-%d').date(),
            origin=data['origin'],
            gender=data['gender'],
            status='ativo',
            herd_id=herd_id,
            user_id=owner_user_id,
            entry_weight=float(entry_weight_value) if entry_weight_value not in [None, '', '0', 0] else None,
            target_weight=float(target_weight_value) if target_weight_value not in [None, '', '0', 0] else None
        )
        
        db.session.add(new_animal)
        db.session.commit()
        
        # Registrar atividade de criação
        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            description = f"Animal criado: {new_animal.earring or new_animal.name or new_animal.id}"
            activity = Activity(
                user_id=int(user_id) if user_id and str(user_id).isdigit() else None,
                username=username,
                action='create',
                object_type='animal',
                object_id=new_animal.id,
                description=description
            )
            db.session.add(activity)
            db.session.commit()
        except Exception as log_error:
            db.session.rollback()
            print(f"DEBUG: Falha ao registrar atividade de criação de gado: {log_error}")

        return make_response(jsonify({
            'message': 'Gado cadastrado com sucesso',
            'animal_id': new_animal.id,
            'data': new_animal.json()
        }), 201)
        
    except Exception as e:
        db.session.rollback()
        return make_response(jsonify({
            'message': f'Erro ao cadastrar gado: {str(e)}'
        }), 500)

@app.route('/api/cattle/<int:cattle_id>', methods=['PUT'])
def update_cattle(cattle_id):
    """Atualizar dados de um gado"""
    try:
        header_user_id = request.headers.get('X-User-Id') or request.headers.get('X-User-ID')
        param_user_id = request.args.get('user_id')
        effective_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            effective_user_id = int(header_user_id)
        elif param_user_id and str(param_user_id).isdigit():
            effective_user_id = int(param_user_id)

        cattle_query = Animal.query.filter_by(id=cattle_id)
        if effective_user_id is not None:
            cattle_query = cattle_query.filter(Animal.user_id == effective_user_id)

        cattle = cattle_query.first()
        if not cattle:
            return make_response(jsonify({'message': 'Gado não encontrado'}), 404)
        
        data = request.json
        
        # Atualizar campos se fornecidos
        if 'name' in data:
            cattle.name = data['name']
        if 'breed' in data:
            cattle.breed = data['breed']
        if 'gender' in data:
            cattle.gender = data['gender']
        if 'origin' in data:
            cattle.origin = data['origin']
        entry_weight_value = data.get('entryWeight', data.get('entry_weight'))
        target_weight_value = data.get('targetWeight', data.get('target_weight'))

        if 'entryWeight' in data or 'entry_weight' in data:
            cattle.entry_weight = float(entry_weight_value) if entry_weight_value not in [None, '', '0', 0] else None
        if 'targetWeight' in data or 'target_weight' in data:
            cattle.target_weight = float(target_weight_value) if target_weight_value not in [None, '', '0', 0] else None
        if 'birthDate' in data:
            from datetime import datetime
            cattle.birth_date = datetime.strptime(data['birthDate'], '%Y-%m-%d').date()
        
        db.session.commit()

        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            description = f"Animal atualizado: {cattle.earring or cattle.name or cattle.id}"
            activity = Activity(
                user_id=int(user_id) if user_id and str(user_id).isdigit() else None,
                username=username,
                action='update',
                object_type='animal',
                object_id=cattle.id,
                description=description
            )
            db.session.add(activity)
            db.session.commit()
        except Exception as log_error:
            db.session.rollback()
            print(f"DEBUG: Falha ao registrar atividade de atualização de gado: {log_error}")

        return make_response(jsonify({
            'message': 'Gado atualizado com sucesso',
            'data': cattle.json()
        }), 200)
        
    except Exception as e:
        db.session.rollback()
        return make_response(jsonify({
            'message': f'Erro ao atualizar gado: {str(e)}'
        }), 500)

@app.route('/api/cattle/<int:cattle_id>', methods=['DELETE', 'OPTIONS'])
def delete_cattle(cattle_id):
    """Deletar um gado"""
    try:
        if request.method == 'OPTIONS':
            return make_response()
            
        print(f"DEBUG: Recebendo requisição DELETE para gado ID: {cattle_id}")
        
        # Importar modelos necessários
        from app.models import Animal, Weighing, Movement, Reproduction, HealthRecord, Attachment
        
        # Verificar se o gado existe
        header_user_id = request.headers.get('X-User-Id')
        payload_user_id = request.json.get('user_id') if request.is_json else None
        effective_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            effective_user_id = int(header_user_id)
        elif payload_user_id and str(payload_user_id).isdigit():
            effective_user_id = int(payload_user_id)

        query = Animal.query.filter_by(id=cattle_id)
        if effective_user_id is not None:
            query = query.filter(Animal.user_id == effective_user_id)

        cattle = query.first()
        if not cattle:
            print(f"DEBUG: Gado com ID {cattle_id} não encontrado")
            return make_response(jsonify({'message': 'Gado não encontrado'}), 404)
        
        print(f"DEBUG: Gado encontrado: {cattle.name}")
        
        # Deletar registros relacionados manualmente primeiro
        try:
            # Deletar pesagens
            Weighing.query.filter_by(animal_id=cattle_id).delete()
            print(f"DEBUG: Pesagens deletadas")
            
            # Deletar movimentos
            Movement.query.filter_by(animal_id=cattle_id).delete()
            print(f"DEBUG: Movimentos deletados")
            
            # Deletar reproduções
            Reproduction.query.filter_by(animal_id=cattle_id).delete()
            print(f"DEBUG: Reproduções deletadas")
            
            # Deletar registros de saúde
            HealthRecord.query.filter_by(animal_id=cattle_id).delete()
            print(f"DEBUG: Registros de saúde deletados")
            
            # Deletar anexos
            Attachment.query.filter_by(animal_id=cattle_id).delete()
            print(f"DEBUG: Anexos deletados")
            
            # Deletar o gado
            db.session.delete(cattle)
            db.session.commit()
            print(f"DEBUG: Gado {cattle_id} deletado com sucesso")
            
        except Exception as delete_error:
            print(f"DEBUG: Erro específico ao deletar: {str(delete_error)}")
            db.session.rollback()
            raise delete_error
        # Registrar atividade de exclusão
        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            description = f"Animal excluído: {cattle.earring or cattle.name or cattle_id}"
            activity = Activity(
                user_id=int(user_id) if user_id and str(user_id).isdigit() else None,
                username=username,
                action='delete',
                object_type='animal',
                object_id=cattle_id,
                description=description
            )
            db.session.add(activity)
            db.session.commit()
        except Exception as log_error:
            db.session.rollback()
            print(f"DEBUG: Falha ao registrar atividade de exclusão de gado: {log_error}")

        return make_response(jsonify({
            'message': 'Gado deletado com sucesso'
        }), 200)
        
    except Exception as e:
        db.session.rollback()
        print(f"DEBUG: Erro ao deletar gado: {str(e)}")
        return make_response(jsonify({
            'message': f'Erro ao deletar gado: {str(e)}'
        }), 500)

@app.route('/api/cattle', methods=['GET'])
def get_cattle():
    """Listar todos os gados"""
    try:
        
        header_user_id = request.headers.get('X-User-Id') or request.headers.get('X-User-ID')
        query_param_user_id = request.args.get('user_id', type=int)
        user_id = None
        if header_user_id and str(header_user_id).isdigit():
            user_id = int(header_user_id)
        elif query_param_user_id:
            user_id = query_param_user_id

        query = Animal.query
        if user_id is not None:
            query = query.filter(Animal.user_id == user_id)
        animals = query.all()
        cattle_list = [animal.json() for animal in animals]
        
        return make_response(jsonify({
            'cattle': cattle_list,
            'total': len(cattle_list)
        }), 200)
        
    except Exception as e:
        return make_response(jsonify({
            'message': f'Erro ao obter lista de gados: {str(e)}'
        }), 500)

# ===== ROTAS PARA ACOMPANHAMENTO DE PESO =====

@app.route('/api/weight', methods=['POST'])
def add_weight():
    try:
        data = request.json
        cattle_id = data.get('cattleId')
        weight = data.get('weight')
        date = data.get('date')
        notes = data.get('notes', '')
        
        if not cattle_id or not weight or not date:
            return make_response(jsonify({'message': 'Dados obrigatórios: cattleId, weight, date'}), 400)
        
        # Verificar se o gado existe
        cattle = Animal.query.get(cattle_id)
        if not cattle:
            return make_response(jsonify({'message': 'Gado não encontrado'}), 404)
        
        # Criar novo registro de pesagem
        from datetime import datetime as dt
        weighing_date = None
        if isinstance(date, str) and date:
            try:
                weighing_date = dt.strptime(date, '%Y-%m-%d').date()
            except ValueError:
                weighing_date = dt.utcnow().date()
        elif isinstance(date, dt):
            weighing_date = date.date()

        weighing = Weighing(
            animal_id=cattle_id,
            weight=float(weight),
            date=weighing_date or dt.utcnow().date(),
            notes=notes
        )
        
        db.session.add(weighing)
        db.session.commit()

        # Registrar atividade de pesagem
        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            description = f"Peso registrado: {float(weight):.0f} Kg para animal {cattle.earring or cattle.name or cattle_id}"
            activity = Activity(
                user_id=int(user_id) if user_id and str(user_id).isdigit() else None,
                username=username,
                action='weigh',
                object_type='weighing',
                object_id=weighing.id,
                description=description
            )
            db.session.add(activity)
            db.session.commit()
        except Exception as log_error:
            db.session.rollback()
            print(f"DEBUG: Falha ao registrar atividade de pesagem: {log_error}")
        
        return make_response(jsonify({
            'message': 'Peso adicionado com sucesso',
            'data': {
                'id': weighing.id,
                'cattle_id': cattle_id,
                'weight': weight,
                'date': date,
                'notes': notes
            }
        }), 200)
    except Exception as e:
        db.session.rollback()
        return make_response(jsonify({'message': f'Erro ao adicionar peso: {str(e)}'}), 500)

@app.route('/api/weight/<int:cattle_id>', methods=['GET'])
def get_weight_history(cattle_id):
    try:
        # Verificar se o gado existe
        cattle = Animal.query.get(cattle_id)
        if not cattle:
            return make_response(jsonify({'message': 'Gado não encontrado'}), 404)
        
        # Buscar histórico de pesagens
        weighings = Weighing.query.filter_by(animal_id=cattle_id).order_by(Weighing.date.desc()).all()
        
        weight_history = []
        for weighing in weighings:
            weight_history.append({
                'id': weighing.id,
                'date': weighing.date.strftime('%d/%m/%Y'),
                'weight': weighing.weight,
                'notes': weighing.notes or ''
            })
        
        return make_response(jsonify({
            'cattle_id': cattle_id,
            'cattle_name': cattle.name,
            'history': weight_history,
            'total_records': len(weight_history)
        }), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao obter histórico: {str(e)}'}), 500)

@app.route('/api/weight/stats', methods=['GET'])
def get_weight_stats():
    try:
        # TODO: Implementar lógica para calcular estatísticas reais
        stats = {
            'averageWeight': 340,
            'maxWeight': 420,
            'minWeight': 280
        }
        return make_response(jsonify(stats), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao obter estatísticas: {str(e)}'}), 500)

@app.route('/api/weight/report', methods=['GET'])
def get_weight_report():
    """Gerar relatório automático de peso baseado nas metas e histórico"""
    try:
        header_user_id = request.headers.get('X-User-Id') or request.headers.get('X-User-ID')
        param_user_id = request.args.get('user_id', type=int)
        effective_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            effective_user_id = int(header_user_id)
        elif param_user_id:
            effective_user_id = param_user_id

        query = Animal.query
        if effective_user_id is not None:
            query = query.filter(Animal.user_id == effective_user_id)

        animals = query.all()

        report_animals = []
        losing_weight_count = 0
        reached_target_count = 0
        with_target_count = 0
        without_data_count = 0
        current_weights = []

        for animal in animals:
            weighings = (
                Weighing.query
                .filter_by(animal_id=animal.id)
                .order_by(Weighing.date.desc())
                .all()
            )

            history = [
                {
                    'id': weighing.id,
                    'date': weighing.date.strftime('%Y-%m-%d') if weighing.date else None,
                    'weight': weighing.weight
                }
                for weighing in weighings[:10]
            ]

            current_weight = None
            last_date = None

            if history:
                current_weight = history[0]['weight']
                last_date = history[0]['date']
            elif animal.entry_weight is not None:
                current_weight = animal.entry_weight
            else:
                without_data_count += 1

            previous_weight = None
            if len(history) > 1:
                previous_weight = history[1]['weight']
            elif animal.entry_weight is not None:
                previous_weight = animal.entry_weight

            entry_weight = animal.entry_weight
            target_weight = animal.target_weight

            weight_change = None
            if current_weight is not None and previous_weight is not None:
                weight_change = round(current_weight - previous_weight, 2)

            difference_to_target = None
            percentage_to_target = None
            status = 'Sem dados suficientes'
            message = 'Cadastre pesagens para obter um acompanhamento preciso.'
            trend = 'stable'

            if current_weight is not None:
                current_weights.append(current_weight)
                status = 'Acompanhamento em progresso'
                message = 'Sem meta definida para este animal.' if not target_weight else ''

                if weight_change is not None:
                    if weight_change > 0.5:
                        trend = 'up'
                    elif weight_change < -0.5:
                        trend = 'down'
                        losing_weight_count += 1
                        status = 'Alerta: perda de peso'
                        message = f'Perdeu {abs(weight_change):.2f} kg desde a última pesagem.'

                if target_weight:
                    with_target_count += 1
                    difference_to_target = round(target_weight - current_weight, 2)

                    if entry_weight is not None:
                        total_needed = target_weight - entry_weight
                        if total_needed > 0:
                            achieved = current_weight - entry_weight
                            percentage_to_target = round(min(max(achieved / total_needed, 0), 1) * 100, 2)

                    if difference_to_target <= 0:
                        reached_target_count += 1
                        status = 'Meta atingida'
                        message = 'Animal já atingiu ou superou o peso de abate.'
                        trend = 'up'
                    elif status != 'Alerta: perda de peso':
                        status = 'Em progresso'
                        message = f'Faltam {abs(difference_to_target):.2f} kg para atingir a meta de abate.'

            report_animals.append({
                'id': animal.id,
                'name': animal.name or animal.earring,
                'currentWeight': current_weight,
                'entryWeight': entry_weight,
                'targetWeight': target_weight,
                'differenceToTarget': difference_to_target,
                'percentageToTarget': percentage_to_target,
                'weightChange': weight_change,
                'lastWeighingDate': last_date,
                'status': status,
                'message': message,
                'trend': trend,
                'history': history
            })

        total_cattle = len(report_animals)
        average_weight = round(sum(current_weights) / len(current_weights), 2) if current_weights else None

        alerts = [animal for animal in report_animals if 'Alerta' in animal['status']]

        report = {
            'summary': {
                'totalCattle': total_cattle,
                'withTarget': sum(1 for a in report_animals if a['targetWeight'] not in (None, '')),
                'reachedTarget': reached_target_count,
                'losingWeight': losing_weight_count,
                'withoutData': without_data_count,
                'averageWeight': average_weight
            },
            'alerts': alerts,
            'animals': report_animals
        }

        return make_response(jsonify(report), 200)
    except Exception as e:
        print(f"DEBUG: Erro ao gerar relatório de peso: {str(e)}")
        return make_response(jsonify({'message': f'Erro ao gerar relatório: {str(e)}'}), 500)

@app.route('/api/cattle/filter', methods=['POST'])
def filter_cattle():
    try:
        filters = request.json or {}
        
        user_id = request.headers.get('X-User-Id') or filters.get('userId')
        herd_id = filters.get('herdId')
        query = Animal.query
        if user_id and str(user_id).isdigit():
            query = query.filter(Animal.user_id == int(user_id))
        if herd_id:
            query = query.filter_by(herd_id=herd_id)

        cattle_list = query.all()
        
        # Filtrar por peso usando o histórico de pesagens
        filtered_cattle = []
        min_weight = filters.get('minWeight', 200)
        max_weight = filters.get('maxWeight', 700)
        breeds_filter = set()
        if isinstance(filters.get('breeds'), dict):
            breeds_filter = {k.lower() for k, v in filters.get('breeds', {}).items() if v}
        elif isinstance(filters.get('breeds'), list):
            breeds_filter = {str(v).lower() for v in filters.get('breeds', [])}

        situations_filter = set()
        if isinstance(filters.get('situations'), dict):
            situations_filter = {k for k, v in filters.get('situations', {}).items() if v}
        elif isinstance(filters.get('situations'), list):
            situations_filter = set(filters.get('situations', []))
        
        for cattle in cattle_list:
            current_weight = 350  # Peso padrão
            situation = "Sem dados"
            
            # Buscar o peso mais recente
            try:
                latest_weighing = Weighing.query.filter_by(animal_id=cattle.id).order_by(Weighing.date.desc()).first()
                if latest_weighing:
                    current_weight = latest_weighing.weight
                    # Calcular situação baseada no peso
                    if current_weight > 400:
                        situation = "Acima da média"
                    elif current_weight < 300:
                        situation = "Abaixo da média"
                    else:
                        situation = "Estável"
                else:
                    # Se não tem pesagem, usar peso de entrada
                    entry_weight = getattr(cattle, 'entry_weight', None)
                    if entry_weight:
                        current_weight = entry_weight
                        situation = "Sem histórico"
                    else:
                        situation = "Sem dados"
            except Exception as e:
                print(f"Erro ao processar gado {cattle.id}: {e}")
                continue
            
            # Verificar se o peso está no range
            if not (min_weight <= current_weight <= max_weight):
                continue

            if breeds_filter:
                breed_normalized = (cattle.breed or '').lower()
                if all(b not in breed_normalized for b in breeds_filter):
                    continue

            if situations_filter and situation:
                normalized_situation = situation.lower()
                matches_situation = False
                situation_map = {
                    'acima da média': 'aboveAverage',
                    'acima da media': 'aboveAverage',
                    'abaixo da média': 'belowAverage',
                    'abaixo da media': 'belowAverage',
                    'crescimento rápido': 'rapidGrowth',
                    'crescimento rapido': 'rapidGrowth',
                    'estável': 'stable',
                    'estavel': 'stable'
                }
                normalized_key = situation_map.get(normalized_situation)
                if normalized_key and normalized_key in situations_filter:
                    matches_situation = True
                elif normalized_situation in situations_filter:
                    matches_situation = True
                if not matches_situation:
                    continue

            filtered_cattle.append({
                'id': cattle.id,
                'name': cattle.name,
                'currentWeight': current_weight,
                'breed': cattle.breed,
                'situation': situation,
                'entry_date': cattle.created_at.strftime('%Y-%m-%d') if cattle.created_at else None,
                'gender': cattle.gender,
                'status': cattle.status
            })
        
        return make_response(jsonify({
            'cattle': filtered_cattle,
            'total': len(filtered_cattle),
            'filters_applied': filters
        }), 200)
    except Exception as e:
        print(f"Erro no filtro: {e}")
        return make_response(jsonify({'message': f'Erro ao filtrar gados: {str(e)}'}), 500)
