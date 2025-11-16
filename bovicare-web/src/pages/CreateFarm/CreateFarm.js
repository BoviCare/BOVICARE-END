import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createHerd, uploadHerdDocuments, getHerds, getHerd, updateHerd, deleteHerd } from '../../services/api';
import Navbar from '../../components/Navbar/Navbar';
import './CreateFarm.css';

const CreateFarm = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  
  const [formData, setFormData] = useState({
    name: '',
    city: '',
    location: '',
    area: '',
    capacity: '',
    ownerName: '',
    employees: '',
    description: ''
  });
  
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [herds, setHerds] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [selectedHerd, setSelectedHerd] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [herdToDelete, setHerdToDelete] = useState(null);

  useEffect(() => {
    loadHerds();
  }, []);

  const resetForm = () => {
    setFormData({
      name: '',
      city: '',
      location: '',
      area: '',
      capacity: '',
      ownerName: '',
      employees: '',
      description: ''
    });
    setAttachedFiles([]);
    setSelectedHerd(null);
  };

  const loadHerds = async () => {
    try {
      const response = await getHerds();
      const herdList = Array.isArray(response) ? response : response?.herds || [];
      setHerds(herdList);
    } catch (err) {
      console.error('Erro ao carregar fazendas:', err);
    }
  };

  const handleStartCreate = () => {
    resetForm();
    setSelectedHerd(null);
    setShowForm(true);
    setError('');
    setSuccess('');
  };

  const handleEditHerd = async (herdId) => {
    try {
      const herd = await getHerd(herdId);
      setSelectedHerd(herd);
      setFormData({
        name: herd.name || '',
        city: herd.city || '',
        location: herd.location || '',
        area: herd.area || '',
        capacity: herd.capacity || '',
        ownerName: herd.owner_name || '',
        employees: herd.employees_count || '',
        description: herd.description || ''
      });
      setShowForm(true); // Changed from setViewMode('edit')
      setError('');
      setSuccess('');
    } catch (err) {
      console.error('Erro ao carregar fazenda:', err);
      setError('Não foi possível carregar os dados da fazenda.');
    }
  };

  const handleDeleteHerd = (herd) => {
    setHerdToDelete(herd);
    setShowDeleteModal(true);
  };

  const confirmDeleteHerd = async () => {
    if (!herdToDelete) return;
    try {
      await deleteHerd(herdToDelete.id);
      if (selectedHerd?.id === herdToDelete.id) {
        resetForm();
      }
      await loadHerds();
      setSuccess('Fazenda excluída com sucesso!');
      setShowForm(false);
    } catch (err) {
      console.error('Erro ao deletar fazenda:', err);
      setError('Erro ao deletar fazenda. Tente novamente.');
    } finally {
      setShowDeleteModal(false);
      setHerdToDelete(null);
    }
  };

  const cancelDeleteHerd = () => {
    setShowDeleteModal(false);
    setHerdToDelete(null);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    setAttachedFiles(prev => [...prev, ...files]);
  };

  const removeFile = (index) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      // Validar campos obrigatórios
      if (!formData.name || !formData.city || !formData.location) {
        throw new Error('Por favor, preencha todos os campos obrigatórios');
      }

      // Preparar dados para envio
      const herdData = {
        name: formData.name,
        description: formData.description || '',
        location: formData.location,
        city: formData.city,
        area: formData.area ? parseFloat(formData.area) : null,
        capacity: formData.capacity ? parseInt(formData.capacity) : null,
        owner_name: formData.ownerName,
        employees_count: formData.employees ? parseInt(formData.employees) : null
      };

      const response = await createHerd(herdData);
      
      // Se há arquivos anexados, fazer upload deles
      if (attachedFiles.length > 0 && response.herd && response.herd.id) {
        try {
          const formData = new FormData();
          attachedFiles.forEach((file) => {
            formData.append('documents', file);
          });
          
          await uploadHerdDocuments(response.herd.id, formData);
        } catch (uploadError) {
          console.warn('Erro ao fazer upload dos documentos:', uploadError);
          // Não falhar a criação da fazenda por causa dos documentos
        }
      }
      
      setSuccess('Fazenda criada com sucesso!');
      resetForm();
      await loadHerds();
      setShowForm(false); // Changed from setViewMode('list')

    } catch (err) {
      setError(err.message || 'Erro ao criar fazenda. Tente novamente.');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    if (!showForm) {
      navigate('/home');
      return;
    }
    setShowForm(false);
    resetForm();
    setError('');
    setSuccess('');
  };

  const handleUpdateHerd = async (e) => {
    e.preventDefault();
    if (!selectedHerd) return;
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const payload = {
        name: formData.name,
        description: formData.description || '',
        location: formData.location,
        city: formData.city,
        area: formData.area ? parseFloat(formData.area) : null,
        capacity: formData.capacity ? parseInt(formData.capacity) : null,
        owner_name: formData.ownerName,
        employees_count: formData.employees ? parseInt(formData.employees) : null
      };

      await updateHerd(selectedHerd.id, payload);
      setSuccess('Fazenda atualizada com sucesso!');
      await loadHerds();
      setShowForm(false);
      resetForm();
    } catch (err) {
      setError(err.message || 'Erro ao atualizar fazenda. Tente novamente.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="create-farm-container">
      <Navbar />

      <div className="create-farm-content">
        <div className="create-farm-header">
          <h1>Gerenciar fazendas</h1>
          <p>Visualize, cadastre e edite suas fazendas cadastradas</p>
        </div>

        {!showForm && (
          <div className="farm-list-section">
            <div className="farm-list-header">
              <h2>Fazendas cadastradas ({herds.length})</h2>
              <button className="add-farm-btn" onClick={handleStartCreate}>
                + Adicionar fazenda
              </button>
            </div>

            {herds.length === 0 ? (
              <div className="farm-empty-state">
                <div className="farm-empty-icon">🏠</div>
                <h3>Nenhuma fazenda cadastrada</h3>
                <p>Cadastre sua primeira fazenda para controlar melhor seu rebanho</p>
                <button className="add-farm-btn" onClick={handleStartCreate}>
                  + Adicionar fazenda
                </button>
              </div>
            ) : (
              <div className="farm-grid">
                {herds.map((herd) => (
                  <div key={herd.id} className="farm-card">
                    <div className="farm-card-header">
                      <h3>{herd.name}</h3>
                      <span>{herd.city || 'Cidade não informada'}</span>
                    </div>

                    <div className="farm-card-body">
                      <div>
                        <span className="info-label">Localização</span>
                        <span className="info-value">{herd.location || '—'}</span>
                      </div>
                      <div>
                        <span className="info-label">Área total</span>
                        <span className="info-value">{herd.area ? `${herd.area} ha` : '—'}</span>
                      </div>
                      <div>
                        <span className="info-label">Capacidade</span>
                        <span className="info-value">{herd.capacity || '—'}</span>
                      </div>
                      <div>
                        <span className="info-label">Funcionários</span>
                        <span className="info-value">{herd.employees_count || '—'}</span>
                      </div>
                    </div>

                    <div className="farm-card-actions">
                      <button
                        type="button"
                        className="edit-farm-btn"
                        onClick={() => handleEditHerd(herd.id)}
                      >
                        Editar
                      </button>
                      <button
                        type="button"
                        className="delete-farm-btn"
                        onClick={() => handleDeleteHerd(herd)}
                      >
                        Excluir
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {showForm && (
          <form onSubmit={selectedHerd ? handleUpdateHerd : handleSubmit} className="create-farm-form">
            <div className="form-fields-grid">
              <div className="form-group">
                <label htmlFor="name">Nome da fazenda *</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  placeholder="Nome da fazenda"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="city">Cidade situada ou próxima *</label>
                <input
                  type="text"
                  id="city"
                  name="city"
                  value={formData.city}
                  onChange={handleInputChange}
                  placeholder="Cidade situada ou próxima"
                  required
                />
              </div>

              <div className="form-group-full">
                <label htmlFor="location">Localização da fazenda *</label>
                <input
                  type="text"
                  id="location"
                  name="location"
                  value={formData.location}
                  onChange={handleInputChange}
                  placeholder="Digite o endereço completo da fazenda"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="area">Área total da fazenda (ha)</label>
                <input
                  type="number"
                  id="area"
                  name="area"
                  value={formData.area}
                  onChange={handleInputChange}
                  placeholder="Área total (hectares)"
                  min="0"
                  step="0.1"
                />
              </div>

              <div className="form-group">
                <label htmlFor="capacity">Capacidade de animais</label>
                <input
                  type="number"
                  id="capacity"
                  name="capacity"
                  value={formData.capacity}
                  onChange={handleInputChange}
                  placeholder="Capacidade de animais"
                  min="0"
                />
              </div>

              <div className="form-group">
                <label htmlFor="ownerName">Nome do proprietário</label>
                <input
                  type="text"
                  id="ownerName"
                  name="ownerName"
                  value={formData.ownerName}
                  onChange={handleInputChange}
                  placeholder="Nome do proprietário"
                />
              </div>

              <div className="form-group">
                <label htmlFor="employees">Número de funcionários</label>
                <input
                  type="number"
                  id="employees"
                  name="employees"
                  value={formData.employees}
                  onChange={handleInputChange}
                  placeholder="Número de funcionários"
                  min="0"
                />
              </div>

              <div className="form-group-full">
                <label htmlFor="description">Descrição da fazenda</label>
                <textarea
                  id="description"
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  placeholder="Descreva as características principais da fazenda"
                  rows="4"
                />
              </div>

              <div className="file-upload-section">
                <label>Anexar documentos e certificações</label>
                <div className="file-upload-row">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="attach-files-btn"
                  >
                    Anexar documentos
                  </button>
                  <span className="file-count">
                    {attachedFiles.length} arquivo(s) anexado(s)
                  </span>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.jpg,.jpeg,.png"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                {attachedFiles.length > 0 && (
                  <div className="attached-files">
                    {attachedFiles.map((file, index) => (
                      <div key={index} className="file-item">
                        <span className="file-name">{file.name}</span>
                        <button
                          type="button"
                          onClick={() => removeFile(index)}
                          className="remove-file-btn"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {error && <div className="error-message">{error}</div>}
              {success && <div className="success-message">{success}</div>}

              <div className="form-actions">
                <button type="button" className="cancel-btn" onClick={handleCancel}>
                  Cancelar
                </button>
                <button type="submit" className="create-btn" disabled={loading}>
                  {loading ? 'Salvando...' : selectedHerd ? 'Salvar alterações' : 'Criar fazenda'}
                </button>
              </div>
            </div>
          </form>
        )}
      </div>

      {showDeleteModal && herdToDelete && (
        <div className="modal-overlay" onClick={cancelDeleteHerd}>
          <div className="modal-content delete-confirmation-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="delete-icon">⚠️</div>
              <h3>Confirmar exclusão</h3>
            </div>
            <div className="delete-message">
              <p>Tem certeza que deseja excluir a fazenda <strong>{herdToDelete.name}</strong>?</p>
              <p className="warning-text">Esta ação não pode ser desfeita e removerá todos os dados associados.</p>
            </div>
            <div className="form-actions">
              <button type="button" className="cancel-btn" onClick={cancelDeleteHerd}>
                Cancelar
              </button>
              <button type="button" className="delete-confirm-btn" onClick={confirmDeleteHerd}>
                Excluir permanentemente
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CreateFarm;
