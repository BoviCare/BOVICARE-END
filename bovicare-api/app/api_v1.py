from flask import request, jsonify, make_response
from app import app, db
from app.models import (
    User, Herd, Animal, Weighing, Movement, Reproduction, 
    Vaccine, VaccineApplication, HealthRecord, Attachment, Activity, UserHerd
)
from sqlalchemy import or_
from datetime import datetime, date
import os
from werkzeug.utils import secure_filename

# ===== ROTAS PARA GESTÃO DE REBANHOS =====

@app.route('/api/v1/herds', methods=['GET'])
def get_herds():
    """Listar todos os rebanhos do usuário"""
    try:
        header_user_id = request.headers.get('X-User-Id') or request.headers.get('X-User-ID')
        query_param_user_id = request.args.get('user_id', type=int)
        user_id = None
        if header_user_id and str(header_user_id).isdigit():
            user_id = int(header_user_id)
        elif query_param_user_id:
            user_id = query_param_user_id

        query = Herd.query
        if user_id is not None:
            query = query.join(UserHerd).filter(UserHerd.user_id == user_id)
        herds = query.all()
        return make_response(jsonify([herd.json() for herd in herds]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar rebanhos: {str(e)}'}), 500)

@app.route('/api/v1/herds', methods=['POST'])
def create_herd():
    """Criar novo rebanho"""
    try:
        data = request.get_json()
        
        if not data.get('name'):
            return make_response(jsonify({'message': 'Nome do rebanho é obrigatório'}), 400)
        
        new_herd = Herd(
            name=data['name'],
            description=data.get('description'),
            location=data.get('location'),
            city=data.get('city'),
            area=data.get('area'),
            capacity=data.get('capacity'),
            owner_name=data.get('owner_name'),
            employees_count=data.get('employees_count')
        )
        
        db.session.add(new_herd)
        db.session.flush()

        user_id = request.headers.get('X-User-Id') or data.get('user_id')
        if not user_id:
            raise ValueError('Usuário não informado para associação da fazenda')
        association = UserHerd(user_id=int(user_id), herd_id=new_herd.id)
        db.session.merge(association)
        
        db.session.commit()
        # Log activity for herd creation
        try:
            username = request.headers.get('X-User-Name')
            act = Activity(
                user_id=owner_user_id,
                username=username,
                action='create',
                object_type='herd',
                object_id=new_herd.id,
                description=f'Rebanho criado: {new_herd.name}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()

        return make_response(jsonify({
            'message': 'Rebanho criado com sucesso',
            'herd': new_herd.json()
        }), 201)
    except Exception as e:
        db.session.rollback()
        return make_response(jsonify({'message': f'Erro ao criar rebanho: {str(e)}'}), 500)

@app.route('/api/v1/herds/<int:herd_id>', methods=['GET'])
def get_herd(herd_id):
    """Buscar rebanho por ID"""
    try:
        user_id = request.args.get('user_id', type=int)
        query = Herd.query.filter_by(id=herd_id)
        if user_id:
            query = query.join(UserHerd).filter(UserHerd.user_id == user_id)
        herd = query.first()
        if herd:
            return make_response(jsonify(herd.json()), 200)
        return make_response(jsonify({'message': 'Rebanho não encontrado'}), 404)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar rebanho: {str(e)}'}), 500)

@app.route('/api/v1/herds/<int:herd_id>', methods=['PUT'])
def update_herd(herd_id):
    """Atualizar rebanho"""
    try:
        user_id = request.headers.get('X-User-Id') or request.args.get('user_id')
        query = Herd.query.filter_by(id=herd_id)
        if user_id and str(user_id).isdigit():
            query = query.join(UserHerd).filter(UserHerd.user_id == int(user_id))
        herd = query.first()
        if not herd:
            return make_response(jsonify({'message': 'Rebanho não encontrado'}), 404)
        
        data = request.get_json()
        herd.name = data.get('name', herd.name)
        herd.description = data.get('description', herd.description)
        herd.location = data.get('location', herd.location)
        herd.city = data.get('city', herd.city)
        herd.area = data.get('area', herd.area)
        herd.capacity = data.get('capacity', herd.capacity)
        herd.owner_name = data.get('owner_name', herd.owner_name)
        herd.employees_count = data.get('employees_count', herd.employees_count)
        herd.updated_at = datetime.utcnow()
        
        db.session.commit()

        try:
            username = request.headers.get('X-User-Name')
            act = Activity(
                user_id=effective_user_id,
                username=username,
                action='update',
                object_type='herd',
                object_id=herd.id,
                description=f'Rebanho atualizado: {herd.name}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()
        
        return make_response(jsonify({
            'message': 'Rebanho atualizado com sucesso',
            'herd': herd.json()
        }), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao atualizar rebanho: {str(e)}'}), 500)

@app.route('/api/v1/herds/<int:herd_id>', methods=['DELETE'])
def delete_herd(herd_id):
    """Deletar rebanho"""
    try:
        user_id = request.headers.get('X-User-Id') or request.args.get('user_id')
        query = Herd.query.filter_by(id=herd_id)
        if user_id and str(user_id).isdigit():
            query = query.join(UserHerd).filter(UserHerd.user_id == int(user_id))
        herd = query.first()
        if not herd:
            return make_response(jsonify({'message': 'Rebanho não encontrado'}), 404)
        
        db.session.delete(herd)
        UserHerd.query.filter_by(herd_id=herd_id).delete()
        db.session.commit()

        try:
            username = request.headers.get('X-User-Name')
            act = Activity(
                user_id=int(user_id) if user_id and str(user_id).isdigit() else None,
                username=username,
                action='delete',
                object_type='herd',
                object_id=herd_id,
                description=f'Rebanho removido: {herd.name}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()
        
        return make_response(jsonify({'message': 'Rebanho deletado com sucesso'}), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao deletar rebanho: {str(e)}'}), 500)

# ===== ROTAS PARA GESTÃO DE ANIMAIS =====

@app.route('/api/v1/animals', methods=['GET'])
def get_animals():
    """Listar todos os animais com filtros"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        herd_id = request.args.get('herd_id', type=int)
        status = request.args.get('status')
        breed = request.args.get('breed')
        user_id_param = request.args.get('user_id')
        header_user_id = request.headers.get('X-User-Id')

        query = Animal.query

        effective_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            effective_user_id = int(header_user_id)
        elif user_id_param and str(user_id_param).isdigit():
            effective_user_id = int(user_id_param)

        if effective_user_id is not None:
            query = query.filter(Animal.user_id == effective_user_id)
        if herd_id:
            query = query.filter_by(herd_id=herd_id)
        if status:
            query = query.filter_by(status=status)
        if breed:
            query = query.filter(Animal.breed.ilike(f'%{breed}%'))
        
        animals = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return make_response(jsonify({
            'animals': [animal.json() for animal in animals.items],
            'total': animals.total,
            'pages': animals.pages,
            'current_page': page
        }), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar animais: {str(e)}'}), 500)

@app.route('/api/v1/animals', methods=['POST'])
def create_animal():
    """Criar novo animal"""
    try:
        data = request.get_json()
        
        header_user_id = request.headers.get('X-User-Id')
        payload_user_id = data.get('user_id')
        owner_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            owner_user_id = int(header_user_id)
        elif payload_user_id and str(payload_user_id).isdigit():
            owner_user_id = int(payload_user_id)

        if owner_user_id is None:
            return make_response(jsonify({'message': 'Usuário não informado para criação do animal'}), 400)

        if not data.get('earring'):
            return make_response(jsonify({'message': 'Brinco é obrigatório'}), 400)
        
        # Verificar se brinco já existe
        existing_animal = Animal.query.filter_by(earring=data['earring']).first()
        if existing_animal:
            return make_response(jsonify({'message': 'Brinco já existe'}), 400)
        
        entry_weight_value = data.get('entry_weight', data.get('entryWeight'))
        target_weight_value = data.get('target_weight', data.get('targetWeight'))

        herd_id = data.get('herd_id')
        if herd_id:
            herd = Herd.query.join(UserHerd).filter(UserHerd.user_id == owner_user_id, Herd.id == herd_id).first()
            if not herd:
                return make_response(jsonify({'message': 'Fazenda não encontrada para o usuário informado'}), 404)

        new_animal = Animal(
            earring=data['earring'],
            name=data.get('name'),
            breed=data.get('breed'),
            birth_date=datetime.strptime(data['birth_date'], '%Y-%m-%d').date() if data.get('birth_date') else None,
            origin=data.get('origin'),
            gender=data.get('gender'),
            status=data.get('status', 'ativo'),
            mother_id=data.get('mother_id'),
            father_id=data.get('father_id'),
            herd_id=herd_id,
            user_id=owner_user_id,
            entry_weight=float(entry_weight_value) if entry_weight_value not in [None, '', '0', 0] else None,
            target_weight=float(target_weight_value) if target_weight_value not in [None, '', '0', 0] else None
        )
        
        db.session.add(new_animal)
        db.session.commit()
        # Log activity
        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            act = Activity(
                user_id=int(user_id) if user_id and user_id.isdigit() else None,
                username=username,
                action='create',
                object_type='animal',
                object_id=new_animal.id,
                description=f'Animal criado: {new_animal.earring} - {new_animal.name}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()
        
        return make_response(jsonify({
            'message': 'Animal criado com sucesso',
            'animal': new_animal.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao criar animal: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>', methods=['GET'])
def get_animal(animal_id):
    """Buscar animal por ID"""
    try:
        user_id_header = request.headers.get('X-User-Id')
        user_id_param = request.args.get('user_id')
        effective_user_id = None
        if user_id_header and str(user_id_header).isdigit():
            effective_user_id = int(user_id_header)
        elif user_id_param and str(user_id_param).isdigit():
            effective_user_id = int(user_id_param)

        query = Animal.query.filter_by(id=animal_id)
        if effective_user_id is not None:
            query = query.filter(Animal.user_id == effective_user_id)

        animal = query.first()
        if animal:
            return make_response(jsonify(animal.json()), 200)
        return make_response(jsonify({'message': 'Animal não encontrado'}), 404)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar animal: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>', methods=['PUT'])
def update_animal(animal_id):
    """Atualizar animal"""
    try:
        header_user_id = request.headers.get('X-User-Id')
        payload_user_id = request.json.get('user_id') if request.is_json else None
        effective_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            effective_user_id = int(header_user_id)
        elif payload_user_id and str(payload_user_id).isdigit():
            effective_user_id = int(payload_user_id)

        query = Animal.query.filter_by(id=animal_id)
        if effective_user_id is not None:
            query = query.filter(Animal.user_id == effective_user_id)

        animal = query.first()
        if not animal:
            return make_response(jsonify({'message': 'Animal não encontrado'}), 404)
        
        data = request.get_json()
        animal.name = data.get('name', animal.name)
        animal.breed = data.get('breed', animal.breed)
        animal.origin = data.get('origin', animal.origin)
        animal.gender = data.get('gender', animal.gender)
        animal.status = data.get('status', animal.status)
        herd_id = data.get('herd_id', animal.herd_id)
        if herd_id:
            herd = Herd.query.join(UserHerd)
            if effective_user_id is not None:
                herd = herd.filter(UserHerd.user_id == effective_user_id)
            herd = herd.filter(Herd.id == herd_id).first()
            if not herd:
                return make_response(jsonify({'message': 'Fazenda não encontrada para o usuário informado'}), 404)
        animal.herd_id = herd_id
        animal.updated_at = datetime.utcnow()
        
        if data.get('birth_date'):
            animal.birth_date = datetime.strptime(data['birth_date'], '%Y-%m-%d').date()

        entry_weight_value = data.get('entry_weight', data.get('entryWeight'))
        target_weight_value = data.get('target_weight', data.get('targetWeight'))

        if 'entry_weight' in data or 'entryWeight' in data:
            animal.entry_weight = float(entry_weight_value) if entry_weight_value not in [None, '', '0', 0] else None
        if 'target_weight' in data or 'targetWeight' in data:
            animal.target_weight = float(target_weight_value) if target_weight_value not in [None, '', '0', 0] else None
        
        db.session.commit()

        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            act = Activity(
                user_id=int(user_id) if user_id and user_id.isdigit() else None,
                username=username,
                action='update',
                object_type='animal',
                object_id=animal.id,
                description=f'Animal atualizado: {animal.earring or animal.name or animal.id}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()
        
        return make_response(jsonify({
            'message': 'Animal atualizado com sucesso',
            'animal': animal.json()
        }), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao atualizar animal: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>', methods=['DELETE'])
def delete_animal(animal_id):
    """Deletar animal"""
    try:
        user_id_header = request.headers.get('X-User-Id')
        user_id_param = request.args.get('user_id')
        effective_user_id = None
        if user_id_header and str(user_id_header).isdigit():
            effective_user_id = int(user_id_header)
        elif user_id_param and str(user_id_param).isdigit():
            effective_user_id = int(user_id_param)

        query = Animal.query.filter_by(id=animal_id)
        if effective_user_id is not None:
            query = query.filter(Animal.user_id == effective_user_id)

        animal = query.first()
        if not animal:
            return make_response(jsonify({'message': 'Animal não encontrado'}), 404)
        
        # Remover dependências antes de excluir o animal
        try:
            Weighing.query.filter_by(animal_id=animal_id).delete()
            Movement.query.filter_by(animal_id=animal_id).delete()
            Reproduction.query.filter_by(animal_id=animal_id).delete()
            HealthRecord.query.filter_by(animal_id=animal_id).delete()
            Attachment.query.filter_by(animal_id=animal_id).delete()
            db.session.flush()
        except Exception as cleanup_error:
            db.session.rollback()
            return make_response(jsonify({'message': f'Erro ao remover dados do animal: {str(cleanup_error)}'}), 500)

        db.session.delete(animal)
        db.session.commit()
        # Log activity
        try:
            act = Activity(
                user_id=None,
                username=None,
                action='delete',
                object_type='animal',
                object_id=animal_id,
                description=f'Animal deletado: id {animal_id}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()
        
        return make_response(jsonify({'message': 'Animal deletado com sucesso'}), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao deletar animal: {str(e)}'}), 500)

# ===== ROTAS PARA PESAGENS =====

@app.route('/api/v1/animals/<int:animal_id>/weighings', methods=['GET'])
def get_animal_weighings(animal_id):
    """Listar pesagens de um animal"""
    try:
        weighings = Weighing.query.filter_by(animal_id=animal_id).order_by(Weighing.date.desc()).all()
        return make_response(jsonify([weighing.json() for weighing in weighings]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar pesagens: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>/weighings', methods=['POST'])
def create_weighing(animal_id):
    """Criar nova pesagem"""
    try:
        data = request.get_json()
        
        if not data.get('weight') or not data.get('date'):
            return make_response(jsonify({'message': 'Peso e data são obrigatórios'}), 400)
        
        new_weighing = Weighing(
            animal_id=animal_id,
            weight=float(data['weight']),
            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
            notes=data.get('notes')
        )
        
        db.session.add(new_weighing)
        db.session.commit()
        # Log activity for weighing
        try:
            user_id = request.headers.get('X-User-Id')
            username = request.headers.get('X-User-Name')
            act = Activity(
                user_id=int(user_id) if user_id and user_id.isdigit() else None,
                username=username,
                action='weigh',
                object_type='weighing',
                object_id=new_weighing.id,
                description=f'Pesagem registrada: {new_weighing.weight} Kg para animal {new_weighing.animal_id}'
            )
            db.session.add(act)
            db.session.commit()
        except Exception:
            db.session.rollback()
        
        return make_response(jsonify({
            'message': 'Pesagem registrada com sucesso',
            'weighing': new_weighing.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao registrar pesagem: {str(e)}'}), 500)

# ===== ROTAS PARA MOVIMENTAÇÕES =====

@app.route('/api/v1/animals/<int:animal_id>/movements', methods=['GET'])
def get_animal_movements(animal_id):
    """Listar movimentações de um animal"""
    try:
        movements = Movement.query.filter_by(animal_id=animal_id).order_by(Movement.date.desc()).all()
        return make_response(jsonify([movement.json() for movement in movements]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar movimentações: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>/movements', methods=['POST'])
def create_movement(animal_id):
    """Criar nova movimentação"""
    try:
        data = request.get_json()
        
        if not data.get('movement_type') or not data.get('date'):
            return make_response(jsonify({'message': 'Tipo de movimentação e data são obrigatórios'}), 400)
        
        new_movement = Movement(
            animal_id=animal_id,
            movement_type=data['movement_type'],
            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
            origin=data.get('origin'),
            destination=data.get('destination'),
            reason=data.get('reason'),
            notes=data.get('notes')
        )
        
        db.session.add(new_movement)
        db.session.commit()
        
        return make_response(jsonify({
            'message': 'Movimentação registrada com sucesso',
            'movement': new_movement.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao registrar movimentação: {str(e)}'}), 500)

# ===== ROTAS PARA REPRODUÇÃO =====

@app.route('/api/v1/animals/<int:animal_id>/reproductions', methods=['GET'])
def get_animal_reproductions(animal_id):
    """Listar reproduções de um animal"""
    try:
        reproductions = Reproduction.query.filter_by(animal_id=animal_id).order_by(Reproduction.date.desc()).all()
        return make_response(jsonify([reproduction.json() for reproduction in reproductions]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar reproduções: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>/reproductions', methods=['POST'])
def create_reproduction(animal_id):
    """Criar nova reprodução"""
    try:
        data = request.get_json()
        
        if not data.get('reproduction_type') or not data.get('date'):
            return make_response(jsonify({'message': 'Tipo de reprodução e data são obrigatórios'}), 400)
        
        new_reproduction = Reproduction(
            animal_id=animal_id,
            reproduction_type=data['reproduction_type'],
            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
            partner_id=data.get('partner_id'),
            expected_birth=datetime.strptime(data['expected_birth'], '%Y-%m-%d').date() if data.get('expected_birth') else None,
            actual_birth=datetime.strptime(data['actual_birth'], '%Y-%m-%d').date() if data.get('actual_birth') else None,
            offspring_id=data.get('offspring_id'),
            success=data.get('success', True),
            notes=data.get('notes')
        )
        
        db.session.add(new_reproduction)
        db.session.commit()
        
        return make_response(jsonify({
            'message': 'Reprodução registrada com sucesso',
            'reproduction': new_reproduction.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao registrar reprodução: {str(e)}'}), 500)

# ===== ROTAS PARA VACINAS =====

@app.route('/api/v1/vaccines', methods=['GET'])
def get_vaccines():
    """Listar todas as vacinas"""
    try:
        vaccines = Vaccine.query.all()
        return make_response(jsonify([vaccine.json() for vaccine in vaccines]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar vacinas: {str(e)}'}), 500)

@app.route('/api/v1/vaccines', methods=['POST'])
def create_vaccine():
    """Criar nova vacina"""
    try:
        data = request.get_json()
        
        if not data.get('name'):
            return make_response(jsonify({'message': 'Nome da vacina é obrigatório'}), 400)
        
        new_vaccine = Vaccine(
            name=data['name'],
            description=data.get('description'),
            manufacturer=data.get('manufacturer'),
            batch_number=data.get('batch_number'),
            expiration_date=datetime.strptime(data['expiration_date'], '%Y-%m-%d').date() if data.get('expiration_date') else None
        )
        
        db.session.add(new_vaccine)
        db.session.commit()
        
        return make_response(jsonify({
            'message': 'Vacina criada com sucesso',
            'vaccine': new_vaccine.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao criar vacina: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>/vaccines', methods=['GET'])
def get_animal_vaccines(animal_id):
    """Listar vacinas aplicadas em um animal"""
    try:
        applications = VaccineApplication.query.filter_by(animal_id=animal_id).order_by(VaccineApplication.application_date.desc()).all()
        return make_response(jsonify([application.json() for application in applications]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar vacinas do animal: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>/vaccines', methods=['POST'])
def apply_vaccine(animal_id):
    """Aplicar vacina em um animal"""
    try:
        data = request.get_json()
        
        if not data.get('vaccine_id') or not data.get('application_date'):
            return make_response(jsonify({'message': 'ID da vacina e data de aplicação são obrigatórios'}), 400)
        
        new_application = VaccineApplication(
            animal_id=animal_id,
            vaccine_id=data['vaccine_id'],
            application_date=datetime.strptime(data['application_date'], '%Y-%m-%d').date(),
            next_dose_date=datetime.strptime(data['next_dose_date'], '%Y-%m-%d').date() if data.get('next_dose_date') else None,
            veterinarian=data.get('veterinarian'),
            notes=data.get('notes')
        )
        
        db.session.add(new_application)
        db.session.commit()
        
        return make_response(jsonify({
            'message': 'Vacina aplicada com sucesso',
            'application': new_application.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao aplicar vacina: {str(e)}'}), 500)

# ===== ROTAS PARA REGISTROS DE SAÚDE =====

@app.route('/api/v1/animals/<int:animal_id>/health', methods=['GET'])
def get_animal_health_records(animal_id):
    """Listar registros de saúde de um animal"""
    try:
        health_records = HealthRecord.query.filter_by(animal_id=animal_id).order_by(HealthRecord.date.desc()).all()
        return make_response(jsonify([record.json() for record in health_records]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar registros de saúde: {str(e)}'}), 500)

@app.route('/api/v1/animals/<int:animal_id>/health', methods=['POST'])
def create_health_record(animal_id):
    """Criar novo registro de saúde"""
    try:
        data = request.get_json()
        
        if not data.get('diagnosis') or not data.get('date'):
            return make_response(jsonify({'message': 'Diagnóstico e data são obrigatórios'}), 400)
        
        new_record = HealthRecord(
            animal_id=animal_id,
            diagnosis=data['diagnosis'],
            treatment=data.get('treatment'),
            veterinarian=data.get('veterinarian'),
            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
            status=data.get('status', 'active'),
            notes=data.get('notes')
        )
        
        db.session.add(new_record)
        db.session.commit()
        
        return make_response(jsonify({
            'message': 'Registro de saúde criado com sucesso',
            'health_record': new_record.json()
        }), 201)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao criar registro de saúde: {str(e)}'}), 500)

# ===== ROTAS PARA ANEXOS =====

@app.route('/api/v1/animals/<int:animal_id>/attachments', methods=['GET'])
def get_animal_attachments(animal_id):
    """Listar anexos de um animal"""
    try:
        attachments = Attachment.query.filter_by(animal_id=animal_id).order_by(Attachment.created_at.desc()).all()
        return make_response(jsonify([attachment.json() for attachment in attachments]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar anexos: {str(e)}'}), 500)

# ===== ROTA DE DASHBOARD =====

@app.route('/api/v1/dashboard', methods=['GET'])
def get_dashboard():
    """Dados para o dashboard"""
    try:
        header_user_id = request.headers.get('X-User-Id') or request.headers.get('X-User-ID')
        query_param_user_id = request.args.get('user_id', type=int)
        effective_user_id = None
        if header_user_id and str(header_user_id).isdigit():
            effective_user_id = int(header_user_id)
        elif query_param_user_id:
            effective_user_id = query_param_user_id

        animal_query = Animal.query
        herd_query = Herd.query
        weighing_query = Weighing.query

        if effective_user_id is not None:
            animal_query = animal_query.filter(Animal.user_id == effective_user_id)
            herd_query = herd_query.join(UserHerd).filter(UserHerd.user_id == effective_user_id)
            weighing_query = weighing_query.join(Animal).filter(Animal.user_id == effective_user_id)

        total_animals = animal_query.count()
        total_herds = herd_query.count()
        active_animals = animal_query.filter_by(status='ativo').count()

        recent_weighings = weighing_query.order_by(Weighing.date.desc()).limit(5).all()

        herds_with_count_query = db.session.query(
            Herd.name,
            db.func.count(Animal.id).label('animal_count')
        ).outerjoin(Animal)

        if effective_user_id is not None:
            herds_with_count_query = herds_with_count_query.join(UserHerd).filter(UserHerd.user_id == effective_user_id)
            herds_with_count_query = herds_with_count_query.filter(db.or_(Animal.user_id == effective_user_id, Animal.user_id.is_(None)))

        herds_with_count = herds_with_count_query.group_by(Herd.id, Herd.name).all()
        
        return make_response(jsonify({
            'total_animals': total_animals,
            'total_herds': total_herds,
            'active_animals': active_animals,
            'recent_weighings': [weighing.json() for weighing in recent_weighings],
            'herds_distribution': [{'name': name, 'count': count} for name, count in herds_with_count]
        }), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar dados do dashboard: {str(e)}'}), 500)


@app.route('/api/v1/activities', methods=['GET'])
def get_activities():
    """Retornar atividades recentes (audit log) permitindo filtro por usuário"""
    try:
        from app.models import Activity

        user_id = request.args.get('user_id', type=int)
        username = request.args.get('username', type=str)
        query = Activity.query

        if user_id and username:
            query = query.filter(or_(Activity.user_id == user_id, Activity.username == username))
        elif user_id:
            query = query.filter(Activity.user_id == user_id)
        elif username:
            query = query.filter(Activity.username == username)

        activities = query.order_by(Activity.created_at.desc()).limit(50).all()
        return make_response(jsonify([a.json() for a in activities]), 200)
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao buscar atividades: {str(e)}'}), 500)

# ===== ROTA PARA UPLOAD DE DOCUMENTOS DE FAZENDA =====

@app.route('/api/v1/herds/<int:herd_id>/documents', methods=['POST'])
def upload_herd_documents(herd_id):
    """Upload de documentos para uma fazenda"""
    try:
        herd = Herd.query.filter_by(id=herd_id).first()
        if not herd:
            return make_response(jsonify({'message': 'Fazenda não encontrada'}), 404)
        
        if 'documents' not in request.files:
            return make_response(jsonify({'message': 'Nenhum arquivo enviado'}), 400)
        
        files = request.files.getlist('documents')
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # Verificar tipo de arquivo
            allowed_extensions = {'pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                continue
            
            # Criar diretório de uploads se não existir
            upload_folder = os.path.join(os.getcwd(), 'uploads', 'herds', str(herd_id))
            os.makedirs(upload_folder, exist_ok=True)
            
            # Gerar nome único para o arquivo
            filename = secure_filename(file.filename)
            unique_filename = f"{int(datetime.now().timestamp())}_{filename}"
            file_path = os.path.join(upload_folder, unique_filename)
            
            # Salvar arquivo
            file.save(file_path)
            
            uploaded_files.append({
                'filename': unique_filename,
                'original_name': file.filename,
                'url': f"/uploads/herds/{herd_id}/{unique_filename}"
            })
        
        return make_response(jsonify({
            'message': f'{len(uploaded_files)} documento(s) enviado(s) com sucesso',
            'files': uploaded_files
        }), 200)
        
    except Exception as e:
        return make_response(jsonify({'message': f'Erro ao fazer upload: {str(e)}'}), 500)
