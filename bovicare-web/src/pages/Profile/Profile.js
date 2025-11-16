import React, { useState, useEffect, useRef } from 'react';
import { FiLock } from 'react-icons/fi';
import Navbar from '../../components/Navbar/Navbar';
import { getCurrentUser, updateProfile, getProfileData, getUserStats, updateProfilePhoto, changePassword } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import './Profile.css';

const Profile = () => {
  const { userId } = useAuth();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isReadyForUpload, setIsReadyForUpload] = useState(false);
  const profileImageRef = useRef(null);

  // Verificar se o componente está pronto para upload
  useEffect(() => {
    if (!loading) {
      const timer = setTimeout(() => {
        const profilePicture = document.querySelector('.profile-picture');
        if (profilePicture) {
          setIsReadyForUpload(true);
        } else {
          setIsReadyForUpload(false);
        }
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [loading]);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showChangePasswordModal, setShowChangePasswordModal] = useState(false);
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [profileData, setProfileData] = useState({
    fullName: '',
    email: '',
    phone: '',
    location: '',
    cpf: '',
    birthDate: '',
    registrationDate: '',
    lastAccess: ''
  });
  const [userStats, setUserStats] = useState({
    farmsRegistered: '',
    cattleRegistered: '',
    subscription: '',
    nextPayment: '',
    nextRenewal: ''
  });
  const [editForm, setEditForm] = useState({
    fullName: '',
    email: '',
    phone: '',
    location: '',
    cpf: '',
    birthDate: ''
  });

  useEffect(() => {
    loadUserData();
  }, []);

  const loadUserData = async () => {
    try {
      setLoading(true);
      
      // Carregar dados em paralelo
      const [userData, profileDataResult, userStatsResult] = await Promise.allSettled([
        getCurrentUser(userId),
        getProfileData(userId),
        getUserStats(userId)
      ]);

      // Atualizar dados do usuário
      if (userData.status === 'fulfilled') {
        const user = userData.value;
        setUser(user);
        
        // Atualizar dados do perfil com dados iniciais do usuário
        setProfileData({
          fullName: user.username || 'Nome não informado',
          email: user.email || 'Email não informado',
          phone: user.phone || '',
          location: user.location || 'Localização não informada',
          cpf: 'CPF não informado',
          birthDate: 'Data de nascimento não informada',
          registrationDate: user.registration_date || 'Data de cadastro não informada',
          lastAccess: 'Último acesso não informado'
        });
        
        setEditForm({
          fullName: user.username || '',
          email: user.email || '',
          phone: user.phone || '',
          location: user.location || '',
          cpf: '',
          birthDate: ''
        });
      } else {
        setUser({ username: 'Caio Silva', email: 'caio@email.com' });
      }

      // Atualizar dados do perfil
      if (profileDataResult.status === 'fulfilled') {
        const profile = profileDataResult.value || {};
        setProfileData(profile);

        // Se vier photo_url, aplicar na imagem ao montar
        try {
          const photoUrl = profile.photo_url || userData.value?.profile_photo_url;
          if (photoUrl && profileImageRef.current) {
            profileImageRef.current.src = `http://localhost:5003${photoUrl}`;
          }
        } catch (e) {
          // ignore
        }
      }

      // Atualizar estatísticas do usuário
      if (userStatsResult.status === 'fulfilled') {
        const stats = userStatsResult.value || {};

        // If cattleRegistered not provided, fallback to counting cattle
        if (!stats.cattleRegistered) {
          try {
            // dynamic import to avoid circular requires
            const { getCattle } = await import('../../services/api');
            const list = await getCattle();
            const total = Array.isArray(list) ? list.length : 0;
            stats.cattleRegistered = `${total}`;
          } catch (e) {
            stats.cattleRegistered = '';
          }
        }

        setUserStats(stats);
      }

    } catch (error) {
      console.error('Erro ao carregar dados do usuário:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleEditProfile = async (e) => {
    e.preventDefault();
    try {
      await updateProfile(editForm);
      
      // Atualizar os dados do perfil localmente
      setProfileData(prev => ({ ...prev, ...editForm }));
      setUser(prev => ({ ...prev, ...editForm }));
      
      setShowEditModal(false);
      
      // Mostrar mensagem bonita
      const successMessage = document.createElement('div');
      successMessage.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.3);
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
      `;
      successMessage.innerHTML = '✓ Perfil atualizado com sucesso!';
      
      // Adicionar animação CSS
      const style = document.createElement('style');
      style.textContent = `
        @keyframes slideIn {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
      `;
      document.head.appendChild(style);
      
      document.body.appendChild(successMessage);
      
      // Remover após 3 segundos
      setTimeout(() => {
        successMessage.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => {
          if (document.body.contains(successMessage)) {
            document.body.removeChild(successMessage);
          }
        }, 300);
      }, 3000);
      
    } catch (error) {
      console.error('Erro ao atualizar perfil:', error);
      
      // Mostrar mensagem de erro bonita
      const errorMessage = document.createElement('div');
      errorMessage.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(244, 67, 54, 0.3);
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
      `;
      errorMessage.innerHTML = '✗ Erro ao atualizar perfil. Tente novamente.';
      
      document.body.appendChild(errorMessage);
      
      setTimeout(() => {
        errorMessage.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => {
          if (document.body.contains(errorMessage)) {
            document.body.removeChild(errorMessage);
          }
        }, 300);
      }, 3000);
    }
  };

  const handlePhotoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Verificar se o componente está pronto para upload
    if (!isReadyForUpload) {
      alert('Aguarde a página carregar completamente antes de fazer upload da foto.');
      return;
    }

    // Validar tipo de arquivo
    if (!file.type.startsWith('image/')) {
      alert('Por favor, selecione apenas arquivos de imagem.');
      return;
    }

    // Validar tamanho (máximo 5MB)
    if (file.size > 5 * 1024 * 1024) {
      alert('A imagem deve ter no máximo 5MB.');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('photo', file);
      
      const response = await updateProfilePhoto(formData);
      
      // Atualizar a imagem usando useRef
      if (response.photo_url && profileImageRef.current) {
        const fullUrl = `http://localhost:5003${response.photo_url}`;
        const imgElement = profileImageRef.current;
        
        imgElement.src = fullUrl;
        
        imgElement.onload = () => {
          // Forçar visibilidade e preenchimento total da imagem
          imgElement.style.setProperty('display', 'block', 'important');
          imgElement.style.setProperty('visibility', 'visible', 'important');
          imgElement.style.setProperty('opacity', '1', 'important');
          imgElement.style.setProperty('width', '100%', 'important');
          imgElement.style.setProperty('height', '100%', 'important');
          imgElement.style.setProperty('object-fit', 'cover', 'important');
          imgElement.style.setProperty('position', 'absolute', 'important');
          imgElement.style.setProperty('top', '0', 'important');
          imgElement.style.setProperty('left', '0', 'important');
          imgElement.style.setProperty('right', '0', 'important');
          imgElement.style.setProperty('bottom', '0', 'important');
          imgElement.style.setProperty('z-index', '1', 'important');
        };
        
        imgElement.onerror = (e) => {
          console.error('Erro ao carregar imagem:', e);
        };
      }
      
   
      
      // Mostrar mensagem bonita
      const successMessage = document.createElement('div');
      successMessage.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.3);
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
      `;
      successMessage.innerHTML = 'Foto atualizada com sucesso!';
      
      // Adicionar animação CSS
      const style = document.createElement('style');
      style.textContent = `
        @keyframes slideIn {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
      `;
      document.head.appendChild(style);
      
      document.body.appendChild(successMessage);
      
      // Remover após 3 segundos
      setTimeout(() => {
        successMessage.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => document.body.removeChild(successMessage), 300);
      }, 3000);
    } catch (error) {
      console.error('Erro ao atualizar foto:', error);
      alert('Erro ao atualizar foto. Tente novamente.');
    }
  };

  const handleChangePassword = async (e) => {
    e.preventDefault();
    
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      alert('As senhas não coincidem');
      return;
    }
    
    if (passwordForm.newPassword.length < 6) {
      alert('A nova senha deve ter pelo menos 6 caracteres');
      return;
    }
    
    try {
      const uid = userId || JSON.parse(localStorage.getItem('user') || '{}')?.id;
      await changePassword({
        current_password: passwordForm.currentPassword,
        new_password: passwordForm.newPassword,
        user_id: uid
      });

      setShowChangePasswordModal(false);
      setPasswordForm({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });

      // Mostrar mensagem de sucesso
      const successMessage = document.createElement('div');
      successMessage.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.3);
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
      `;
      successMessage.innerHTML = '✓ Senha alterada com sucesso!';
      document.body.appendChild(successMessage);

      setTimeout(() => {
        successMessage.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => document.body.removeChild(successMessage), 300);
      }, 3000);

    } catch (err) {
      console.error('Erro ao alterar senha:', err);
      // Tentar extrair mensagem do axios error
      const msg = err.response?.data?.message || err.message || 'Erro ao alterar senha';
      alert(msg);
    }
  };

  if (loading) {
    return (
      <div className="profile-container">
        <Navbar />
        <div className="profile-loading">
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="profile-container">
      <Navbar />
      <div className="profile-content">
        <h1 className="profile-title">Perfil</h1>
        
        {/* Single Profile Card */}
        <div className="profile-card">
          {/* Header Section */}
          <div className="profile-header">
            <div className="profile-picture-section">
              <div className="profile-picture">
                <img ref={profileImageRef} src="" alt="" />
                <button 
                  className="upload-photo-btn"
                  onClick={() => document.getElementById('photo-upload').click()}
                  disabled={!isReadyForUpload}
                  style={{ 
                    opacity: isReadyForUpload ? 1 : 0.5,
                    cursor: isReadyForUpload ? 'pointer' : 'not-allowed'
                  }}
                >
                  +
                </button>
                <input
                  id="photo-upload"
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={handlePhotoUpload}
                />
              </div>
              <div className="profile-info">
                <h2 className="profile-name">{profileData.fullName || user?.username || 'Nome não informado'}</h2>
                <p className="profile-location">{profileData.location || user?.location || 'Localização não informada'} • Usuário desde {profileData.registrationDate || 'data não informada'}</p>
              </div>
            </div>
            <div className="profile-buttons">
              <button 
                className="edit-profile-btn"
                onClick={() => {
                  // Carregar dados atuais no formulário
                  setEditForm({
                    fullName: profileData.fullName || '',
                    email: profileData.email || '',
                    phone: profileData.phone || '',
                    location: profileData.location || '',
                    cpf: profileData.cpf || '',
                    birthDate: profileData.birthDate || ''
                  });
                  setShowEditModal(true);
                }}
              >
                Editar perfil
              </button>
               <button 
                 className="edit-profile-btn"
                 onClick={() => setShowChangePasswordModal(true)}
                 style={{
                   background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
                   color: 'white',
                   border: 'none',
                   display: 'flex',
                   alignItems: 'center',
                   justifyContent: 'center',
                   gap: '6px'
                 }}
               >
                 <FiLock size={16} />
                 Alterar Senha
               </button>
            </div>
          </div>

          {/* Details Section - Two columns inside the same card */}
          <div className="profile-details">
            <div className="profile-column">
              <div className="profile-field">
                <label>Nome completo</label>
                <span>{profileData.fullName}</span>
              </div>
              <div className="profile-field">
                <label>Data de nascimento</label>
                <span>{profileData.birthDate}</span>
              </div>
              <div className="profile-field">
                <label>Localização</label>
                <span>{profileData.location}</span>
              </div>
              <div className="profile-field">
                <label>Email</label>
                <span>{profileData.email}</span>
              </div>
              <div className="profile-field">
                <label>Telefone</label>
                <span>{profileData.phone}</span>
              </div>
              <div className="profile-field">
                <label>Fazendas cadastradas</label>
                <span>{userStats.farmsRegistered}</span>
              </div>
              <div className="profile-field">
                <label>Gados cadastrados</label>
                <span>{userStats.cattleRegistered}</span>
              </div>
              <div className="profile-field">
                <label>Data de cadastro</label>
                <span>{profileData.registrationDate}</span>
              </div>
              <div className="profile-field">
                <label>Último acesso</label>
                <span>{profileData.lastAccess}</span>
              </div>
            </div>

            <div className="profile-column">
              <div className="profile-field">
                <label>CPF</label>
                <span>{profileData.cpf}</span>
              </div>
              <div className="profile-field">
                <label>Assinatura</label>
                <span className="subscription-badge">{userStats.subscription}</span>
              </div>
              <div className="profile-field">
                <label>Próximo pagamento</label>
                <span>{userStats.nextPayment}</span>
              </div>
              <div className="profile-field">
                <label>Próxima renovação da assinatura</label>
                <span>{userStats.nextRenewal}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Edit Profile Modal */}
      {showEditModal && (
        <div className="profile-modal">
          <div className="profile-modal-content">
            <div className="modal-header">
              <h3>Editar Perfil</h3>
              <button 
                className="close-btn"
                onClick={() => setShowEditModal(false)}
              >
                ×
              </button>
            </div>
            
            <form onSubmit={handleEditProfile} className="profile-form">
              <div className="form-group">
                <label>Nome completo</label>
                <input
                  type="text"
                  value={editForm.fullName}
                  onChange={(e) => setEditForm({...editForm, fullName: e.target.value})}
                  required
                />
              </div>
              
              <div className="form-group">
                <label>Email</label>
                <input
                  type="email"
                  value={editForm.email}
                  onChange={(e) => setEditForm({...editForm, email: e.target.value})}
                  required
                />
              </div>
              
              <div className="form-group">
                <label>Telefone</label>
                <input
                  type="tel"
                  value={editForm.phone}
                  onChange={(e) => setEditForm({...editForm, phone: e.target.value})}
                />
              </div>
              
              <div className="form-group">
                <label>Localização</label>
                <input
                  type="text"
                  value={editForm.location}
                  onChange={(e) => setEditForm({...editForm, location: e.target.value})}
                />
              </div>
              
              <div className="form-group">
                <label>CPF</label>
                <input
                  type="text"
                  value={editForm.cpf}
                  onChange={(e) => setEditForm({...editForm, cpf: e.target.value})}
                />
              </div>
              
              <div className="form-group">
                <label>Data de nascimento</label>
                <input
                  type="date"
                  value={editForm.birthDate}
                  onChange={(e) => setEditForm({...editForm, birthDate: e.target.value})}
                />
              </div>
              
              <div className="form-actions">
                <button 
                  type="button" 
                  className="cancel-btn"
                  onClick={() => setShowEditModal(false)}
                >
                  Cancelar
                </button>
                <button type="submit" className="save-btn">
                  Salvar
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal Alterar Senha */}
      {showChangePasswordModal && (
        <div className="profile-modal">
          <div className="profile-modal-content">
            <div className="modal-header">
              <h3>Alterar Senha</h3>
              <button 
                className="close-btn"
                onClick={() => setShowChangePasswordModal(false)}
              >
                ×
              </button>
            </div>
            
            <form onSubmit={handleChangePassword} className="profile-form">
              <div className="form-group">
                <label>Senha Atual *</label>
                <input
                  type="password"
                  value={passwordForm.currentPassword}
                  onChange={(e) => setPasswordForm({...passwordForm, currentPassword: e.target.value})}
                  required
                  placeholder="Digite sua senha atual"
                />
              </div>
              
              <div className="form-group">
                <label>Nova Senha *</label>
                <input
                  type="password"
                  value={passwordForm.newPassword}
                  onChange={(e) => setPasswordForm({...passwordForm, newPassword: e.target.value})}
                  required
                  placeholder="Digite sua nova senha (mínimo 6 caracteres)"
                  minLength="6"
                />
              </div>
              
              <div className="form-group">
                <label>Confirmar Nova Senha *</label>
                <input
                  type="password"
                  value={passwordForm.confirmPassword}
                  onChange={(e) => setPasswordForm({...passwordForm, confirmPassword: e.target.value})}
                  required
                  placeholder="Confirme sua nova senha"
                  minLength="6"
                />
              </div>
              
              <div className="form-actions">
                <button 
                  type="button" 
                  className="cancel-btn"
                  onClick={() => setShowChangePasswordModal(false)}
                >
                  Cancelar
                </button>
                <button type="submit" className="save-btn">
                  Alterar Senha
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Profile;
