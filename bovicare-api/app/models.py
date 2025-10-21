from app import db
from datetime import datetime, timedelta
import secrets
from enum import Enum

# Enums para padronização
class AnimalStatus(Enum):
    ATIVO = "ativo"
    VENDIDO = "vendido"
    MORTO = "morto"
    TRANSFERIDO = "transferido"

class MovementType(Enum):
    ENTRADA = "entrada"
    SAIDA = "saida"
    TRANSFERENCIA = "transferencia"

class ReproductionType(Enum):
    COBERTURA_NATURAL = "cobertura_natural"
    INSEMINACAO_ARTIFICIAL = "inseminacao_artificial"
    TRANSFERENCIA_EMBRIAO = "transferencia_embriao"

class UserHerd(db.Model):
    __tablename__ = 'user_herds'

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    herd_id = db.Column(db.Integer, db.ForeignKey('herds.id'), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Modelo de Usuário (expandido)
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), default='user')  # admin, veterinarian, technician, user
    is_active = db.Column(db.Boolean, default=True)
    profile_photo_url = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    herds = db.relationship('Herd', secondary='user_herds', back_populates='owners')

    def json(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'phone': self.phone,
            'role': self.role,
            'is_active': self.is_active,
            'profile_photo_url': getattr(self, 'profile_photo_url', None),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

class PasswordReset(db.Model):
    __tablename__ = 'password_resets'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    method = db.Column(db.String(10), nullable=False)  # 'email' ou 'sms'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    
    def __init__(self, user_id, method):
        self.user_id = user_id
        self.code = self.generate_code()
        self.method = method
        self.expires_at = datetime.utcnow() + timedelta(minutes=30)  # Código expira em 30 minutos
    
    def generate_code(self):
        return ''.join([str(secrets.randbelow(10)) for _ in range(6)])
    
    def is_valid(self):
        return not self.used and datetime.utcnow() < self.expires_at
    
    def json(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'code': self.code,
            'method': self.method,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'used': self.used
        }

# ===== MODELOS PARA GESTÃO DE GADO =====

# Rebanhos/Lotes/Fazendas
class Herd(db.Model):
    __tablename__ = 'herds'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    location = db.Column(db.String(200))
    city = db.Column(db.String(100))  # Cidade
    area = db.Column(db.Float)  # Área em hectares
    capacity = db.Column(db.Integer)
    owner_name = db.Column(db.String(100))  # Nome do proprietário
    employees_count = db.Column(db.Integer)  # Número de funcionários
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owners = db.relationship('User', secondary='user_herds', back_populates='herds')
    animals = db.relationship('Animal', backref='herd', lazy=True)
    
    def json(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'location': self.location,
            'city': self.city,
            'area': self.area,
            'capacity': self.capacity,
            'owner_name': self.owner_name,
            'employees_count': self.employees_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Animais
class Animal(db.Model):
    __tablename__ = 'animals'
    
    id = db.Column(db.Integer, primary_key=True)
    earring = db.Column(db.String(20), unique=True, nullable=False)  # Brinco
    name = db.Column(db.String(100))
    breed = db.Column(db.String(50))  # Raça
    birth_date = db.Column(db.Date)
    origin = db.Column(db.String(100))  # Origem
    gender = db.Column(db.String(10))  # M/F
    status = db.Column(db.String(20), default=AnimalStatus.ATIVO.value)
    entry_weight = db.Column(db.Float, nullable=True)
    target_weight = db.Column(db.Float, nullable=True)
    mother_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='SET NULL'))
    father_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='SET NULL'))
    herd_id = db.Column(db.Integer, db.ForeignKey('herds.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = db.relationship('User', backref=db.backref('animals', lazy=True))
    
    def json(self):
        return {
            'id': self.id,
            'earring': self.earring,
            'name': self.name,
            'breed': self.breed,
            'birth_date': self.birth_date.isoformat() if self.birth_date else None,
            'origin': self.origin,
            'gender': self.gender,
            'status': self.status,
            'entry_weight': self.entry_weight,
            'target_weight': self.target_weight,
            'mother_id': self.mother_id,
            'father_id': self.father_id,
            'herd_id': self.herd_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Pesagens
class Weighing(db.Model):
    __tablename__ = 'weighings'
    
    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='CASCADE'), nullable=False)
    weight = db.Column(db.Float, nullable=False)  # Peso em kg
    date = db.Column(db.Date, nullable=False)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def json(self):
        return {
            'id': self.id,
            'animal_id': self.animal_id,
            'weight': self.weight,
            'date': self.date.isoformat() if self.date else None,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# Modelo para atividades / audit log
class Activity(db.Model):
    __tablename__ = 'activities'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    username = db.Column(db.String(100), nullable=True)
    action = db.Column(db.String(50), nullable=False)  # create, update, delete, weigh
    object_type = db.Column(db.String(50), nullable=True)  # animal, weighing, herd, user
    object_id = db.Column(db.Integer, nullable=True)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def _derive_icon(self):
        icon_map = {
            'animal': 'vaca',
            'weighing': 'vaca',
            'herd': 'cadastrargado',
            'user': 'Cadastro',
        }
        if self.object_type and self.object_type.lower() in icon_map:
            return icon_map[self.object_type.lower()]
        if self.action and self.action.lower() in icon_map:
            return icon_map[self.action.lower()]
        return 'vaca'

    def json(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.username,
            'action': self.action,
            'object_type': self.object_type,
            'object_id': self.object_id,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'icon': self._derive_icon(),
            'type': self.object_type or self.action
        }

# Movimentações
class Movement(db.Model):
    __tablename__ = 'movements'
    
    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='CASCADE'), nullable=False)
    movement_type = db.Column(db.String(20), nullable=False)  # entrada, saida, transferencia
    date = db.Column(db.Date, nullable=False)
    origin = db.Column(db.String(200))
    destination = db.Column(db.String(200))
    reason = db.Column(db.String(200))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def json(self):
        return {
            'id': self.id,
            'animal_id': self.animal_id,
            'movement_type': self.movement_type,
            'date': self.date.isoformat() if self.date else None,
            'origin': self.origin,
            'destination': self.destination,
            'reason': self.reason,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Reprodução
class Reproduction(db.Model):
    __tablename__ = 'reproductions'
    
    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='CASCADE'), nullable=False)
    reproduction_type = db.Column(db.String(30), nullable=False)  # cobertura_natural, inseminacao_artificial, etc.
    date = db.Column(db.Date, nullable=False)
    partner_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='SET NULL'))  # Parceiro (touro)
    expected_birth = db.Column(db.Date)  # Data esperada do parto
    actual_birth = db.Column(db.Date)  # Data real do parto
    offspring_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='SET NULL'))  # Filhote gerado
    success = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def json(self):
        return {
            'id': self.id,
            'animal_id': self.animal_id,
            'reproduction_type': self.reproduction_type,
            'date': self.date.isoformat() if self.date else None,
            'partner_id': self.partner_id,
            'expected_birth': self.expected_birth.isoformat() if self.expected_birth else None,
            'actual_birth': self.actual_birth.isoformat() if self.actual_birth else None,
            'offspring_id': self.offspring_id,
            'success': self.success,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Vacinas e Protocolos
class Vaccine(db.Model):
    __tablename__ = 'vaccines'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    manufacturer = db.Column(db.String(100))
    batch_number = db.Column(db.String(50))
    expiration_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relacionamentos
    vaccine_applications = db.relationship('VaccineApplication', backref='vaccine', lazy=True)
    
    def json(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'manufacturer': self.manufacturer,
            'batch_number': self.batch_number,
            'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class VaccineApplication(db.Model):
    __tablename__ = 'vaccine_applications'
    
    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='CASCADE'), nullable=False)
    vaccine_id = db.Column(db.Integer, db.ForeignKey('vaccines.id'), nullable=False)
    application_date = db.Column(db.Date, nullable=False)
    next_dose_date = db.Column(db.Date)
    veterinarian = db.Column(db.String(100))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def json(self):
        return {
            'id': self.id,
            'animal_id': self.animal_id,
            'vaccine_id': self.vaccine_id,
            'application_date': self.application_date.isoformat() if self.application_date else None,
            'next_dose_date': self.next_dose_date.isoformat() if self.next_dose_date else None,
            'veterinarian': self.veterinarian,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Registros de Saúde
class HealthRecord(db.Model):
    __tablename__ = 'health_records'
    
    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='CASCADE'), nullable=False)
    diagnosis = db.Column(db.String(200), nullable=False)
    treatment = db.Column(db.Text)
    veterinarian = db.Column(db.String(100))
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), default='active')  # active, resolved, ongoing
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def json(self):
        return {
            'id': self.id,
            'animal_id': self.animal_id,
            'diagnosis': self.diagnosis,
            'treatment': self.treatment,
            'veterinarian': self.veterinarian,
            'date': self.date.isoformat() if self.date else None,
            'status': self.status,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Anexos e Documentos
class Attachment(db.Model):
    __tablename__ = 'attachments'
    
    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey('animals.id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_type = db.Column(db.String(50))  # image, document, etc.
    file_size = db.Column(db.Integer)  # em bytes
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def json(self):
        return {
            'id': self.id,
            'animal_id': self.animal_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }