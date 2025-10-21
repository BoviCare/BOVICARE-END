import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { getUsers, createUser, updateUser, deleteUser } from '../../services/api';
import Navbar from '../../components/Navbar/Navbar';
import './Users.css';

const Users = () => {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');

  // Verificar se o usuário atual é admin
  const isAdmin = currentUser?.role === 'admin';

  useEffect(() => {
    if (isAdmin) {
      loadUsers();
    }
  }, [isAdmin]);

  const loadUsers = async () => {
    try {
      setLoading(true);
      const response = await getUsers();
      setUsers(response);
    } catch (err) {
      setError('Erro ao carregar usuários: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateUser = async (userData) => {
    try {
      await createUser(userData);
      setSuccess('Usuário criado com sucesso!');
      setShowCreateModal(false);
      loadUsers();
    } catch (err) {
      setError('Erro ao criar usuário: ' + err.message);
    }
  };

  const handleUpdateUser = async (userId, userData) => {
    try {
      await updateUser(userId, userData);
      setSuccess('Usuário atualizado com sucesso!');
      setShowEditModal(false);
      setSelectedUser(null);
      loadUsers();
    } catch (err) {
      setError('Erro ao atualizar usuário: ' + err.message);
    }
  };

  const handleDeleteUser = async (userId) => {
    if (window.confirm('Tem certeza que deseja excluir este usuário?')) {
      try {
        await deleteUser(userId);
        setSuccess('Usuário excluído com sucesso!');
        loadUsers();
      } catch (err) {
        setError('Erro ao excluir usuário: ' + err.message);
      }
    }
  };

  const handleEditUser = (user) => {
    setSelectedUser(user);
    setShowEditModal(true);
  };

  // Filtrar usuários
  const filteredUsers = users.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = roleFilter === 'all' || user.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  // Se não for admin, mostrar mensagem de acesso negado
  if (!isAdmin) {
    return (
      <div className="users-container">
        <Navbar />
        <div className="access-denied">
          <h2>🚫 Acesso Negado</h2>
          <p>Você não tem permissão para acessar esta página.</p>
          <p>Apenas administradores podem gerenciar usuários.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="users-container">
      <Navbar />
      
      <div className="users-content">
        <div className="users-header">
          <h1>Gerenciar Usuários</h1>
          <button 
            className="create-user-btn"
            onClick={() => setShowCreateModal(true)}
          >
            + Novo Usuário
          </button>
        </div>

        {/* Filtros */}
        <div className="users-filters">
          <div className="search-box">
            <input
              type="text"
              placeholder="Buscar usuários..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="role-filter">
            <select
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value)}
            >
              <option value="all">Todos os roles</option>
              <option value="admin">Admin</option>
              <option value="user">User</option>
              <option value="veterinarian">Veterinário</option>
              <option value="technician">Técnico</option>
            </select>
          </div>
        </div>

        {/* Mensagens */}
        {error && (
          <div className="error-message">
            {error}
            <button onClick={() => setError('')}>×</button>
          </div>
        )}

        {success && (
          <div className="success-message">
            {success}
            <button onClick={() => setSuccess('')}>×</button>
          </div>
        )}

        {/* Lista de usuários */}
        <div className="users-list">
          {loading ? (
            <div className="loading">Carregando usuários...</div>
          ) : filteredUsers.length === 0 ? (
            <div className="no-users">
              <p>Nenhum usuário encontrado.</p>
            </div>
          ) : (
            <div className="users-grid">
              {filteredUsers.map(user => (
                <div key={user.id} className="user-card">
                  <div className="user-info">
                    <div className="user-avatar">
                      {user.username.charAt(0).toUpperCase()}
                    </div>
                    <div className="user-details">
                      <h3>{user.username}</h3>
                      <p className="user-email">{user.email}</p>
                      <p className="user-phone">{user.phone || 'Sem telefone'}</p>
                      <span className={`user-role role-${user.role}`}>
                        {user.role}
                      </span>
                    </div>
                  </div>
                  <div className="user-actions">
                    <button 
                      className="edit-btn"
                      onClick={() => handleEditUser(user)}
                    >
                      Editar
                    </button>
                    <button 
                      className="delete-btn"
                      onClick={() => handleDeleteUser(user.id)}
                      disabled={user.id === currentUser.id}
                    >
                      Excluir
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modal de criação de usuário */}
      {showCreateModal && (
        <CreateUserModal
          onClose={() => setShowCreateModal(false)}
          onSubmit={handleCreateUser}
        />
      )}

      {/* Modal de edição de usuário */}
      {showEditModal && selectedUser && (
        <EditUserModal
          user={selectedUser}
          onClose={() => {
            setShowEditModal(false);
            setSelectedUser(null);
          }}
          onSubmit={handleUpdateUser}
        />
      )}
    </div>
  );
};

// Componente para modal de criação de usuário
const CreateUserModal = ({ onClose, onSubmit }) => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    phone: '',
    role: 'user'
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Criar Novo Usuário</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        <form onSubmit={handleSubmit} className="user-form">
          <div className="form-group">
            <label>Nome de usuário *</label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Email *</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Senha *</label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Telefone</label>
            <input
              type="tel"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label>Role *</label>
            <select
              name="role"
              value={formData.role}
              onChange={handleChange}
              required
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
              <option value="veterinarian">Veterinário</option>
              <option value="technician">Técnico</option>
            </select>
          </div>
          <div className="form-actions">
            <button type="button" onClick={onClose} className="cancel-btn">
              Cancelar
            </button>
            <button type="submit" className="submit-btn">
              Criar Usuário
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// Componente para modal de edição de usuário
const EditUserModal = ({ user, onClose, onSubmit }) => {
  const [formData, setFormData] = useState({
    username: user.username,
    email: user.email,
    phone: user.phone || '',
    role: user.role
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(user.id, formData);
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Editar Usuário</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        <form onSubmit={handleSubmit} className="user-form">
          <div className="form-group">
            <label>Nome de usuário *</label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Email *</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label>Telefone</label>
            <input
              type="tel"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label>Role *</label>
            <select
              name="role"
              value={formData.role}
              onChange={handleChange}
              required
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
              <option value="veterinarian">Veterinário</option>
              <option value="technician">Técnico</option>
            </select>
          </div>
          <div className="form-actions">
            <button type="button" onClick={onClose} className="cancel-btn">
              Cancelar
            </button>
            <button type="submit" className="submit-btn">
              Salvar Alterações
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Users;
