import React, { useState, useEffect } from 'react';
import Navbar from '../../components/Navbar/Navbar';
import { addCattle, getCattle, getHerds, deleteCattle, updateCattle } from '../../services/api';
import './CadastrarGado.css';

const CadastrarGado = () => {
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [cattleList, setCattleList] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [listLoading, setListLoading] = useState(true);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedCattle, setSelectedCattle] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [cattleToDelete, setCattleToDelete] = useState(null);

  // Estado do formulário
  const [formData, setFormData] = useState({
    name: '',                    // Nome ou código interno do animal
    entryDate: '',              // Data de entrada
    herdId: '',                 // ID da fazenda/rebanho selecionado
    gender: '',                 // Sexo
    breed: '',                  // Raça
    category: '',               // Categoria animal
    customCategory: '',         // Categoria customizada (quando "outros")
    entryWeight: '',            // Peso de entrada
    birthDate: '',              // Data de nascimento (para calcular idade)
    targetWeight: '',           // Peso de meta para saída
    estimatedSlaughter: ''     // Previsão de abate
  });

  // Opções para os selects
  const genderOptions = [
    { value: 'macho', label: 'Macho' },
    { value: 'femea', label: 'Fêmea' },
    { value: 'castrado', label: 'Castrado' }
  ];

  const categoryOptions = [
    { value: 'bezerro', label: 'Bezerro' },
    { value: 'garrote', label: 'Garrote' },
    { value: 'novilho', label: 'Novilho' },
    { value: 'vaca', label: 'Vaca' },
    { value: 'touro', label: 'Touro' },
    { value: 'outros', label: 'Outros' }
  ];

  // Carregar lista de gados ao montar o componente
  useEffect(() => {
    loadCattleList();
    loadHerds();
  }, []);

  const [herds, setHerds] = useState([]);
  const [herdsLoading, setHerdsLoading] = useState(true);
  // Notification state for toasts
  const [notification, setNotification] = useState({ visible: false, type: 'success', message: '' });

  const loadHerds = async () => {
    try {
      setHerdsLoading(true);
      const response = await getHerds();
      // response may be an array or wrapped object depending on backend
      const list = Array.isArray(response) ? response : (response.herds || response);
      setHerds(list);
    } catch (error) {
      console.error('Erro ao carregar rebanhos:', error);
    } finally {
      setHerdsLoading(false);
    }
  };

  // Função para carregar lista de gados
  const loadCattleList = async () => {
    try {
      setListLoading(true);
      const response = await getCattle();
      // getCattle() returns an array of animals (api v1 returns { animals: [...] } handled in service)
      // but the service currently returns the array itself. Accept multiple shapes defensively.
      if (Array.isArray(response)) {
        setCattleList(response);
      } else if (response && Array.isArray(response.animals)) {
        setCattleList(response.animals);
      } else if (response && Array.isArray(response.cattle)) {
        setCattleList(response.cattle);
      } else {
        setCattleList([]);
      }
    } catch (error) {
      console.error('Erro ao carregar lista de gados:', error);
      setErrorMessage('Erro ao carregar lista de gados');
    } finally {
      setListLoading(false);
    }
  };

  // Função para lidar com mudanças nos campos
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Limpar mensagens ao digitar
    setSuccessMessage('');
    setErrorMessage('');
  };

  // Função para calcular idade baseada na data de nascimento
  const calculateAge = (birthDate) => {
    if (!birthDate) return '';
    
    const today = new Date();
    const birth = new Date(birthDate);
    const diffTime = Math.abs(today - birth);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 30) {
      return `${diffDays} dias`;
    } else if (diffDays < 365) {
      const months = Math.floor(diffDays / 30);
      return `${months} meses`;
    } else {
      const years = Math.floor(diffDays / 365);
      const months = Math.floor((diffDays % 365) / 30);
      return `${years} anos e ${months} meses`;
    }
  };

  // Função para validar o formulário
  const validateForm = () => {
    const requiredFields = [
      'name', 'entryDate', 'herdId', 'gender', 'breed',
      'category', 'entryWeight', 'birthDate', 'targetWeight', 'estimatedSlaughter'
    ];

    for (const field of requiredFields) {
      const value = formData[field];
      const empty = value === undefined || value === null || (typeof value === 'string' && value.trim() === '');
      if (empty) {
        setErrorMessage(`Campo obrigatório não preenchido: ${getFieldLabel(field)}`);
        return false;
      }
    }

    // Validar se categoria é "outros" e campo customizado está preenchido
    if (formData.category === 'outros' && !formData.customCategory.trim()) {
      setErrorMessage('Por favor, especifique a categoria personalizada.');
      return false;
    }

    // Validar datas
    const entryDate = new Date(formData.entryDate);
    const birthDate = new Date(formData.birthDate);
    const slaughterDate = new Date(formData.estimatedSlaughter);

    if (birthDate > entryDate) {
      setErrorMessage('A data de nascimento não pode ser posterior à data de entrada.');
      return false;
    }

    if (slaughterDate <= entryDate) {
      setErrorMessage('A previsão de abate deve ser posterior à data de entrada.');
      return false;
    }

    // Validar pesos
    const entryWeight = parseFloat(formData.entryWeight);
    const targetWeight = parseFloat(formData.targetWeight);

    if (entryWeight <= 0) {
      setErrorMessage('O peso de entrada deve ser maior que zero.');
      return false;
    }

    if (targetWeight <= entryWeight) {
      setErrorMessage('O peso de meta deve ser maior que o peso de entrada.');
      return false;
    }

    return true;
  };

  // Função para obter o label do campo
  const getFieldLabel = (field) => {
    const labels = {
      name: 'Nome ou código interno',
  entryDate: 'Data de entrada',
      herdId: 'Fazenda / Rebanho',
      gender: 'Sexo',
      breed: 'Raça',
      category: 'Categoria animal',
      entryWeight: 'Peso de entrada',
      birthDate: 'Data de nascimento',
      targetWeight: 'Peso de meta para saída',
      estimatedSlaughter: 'Previsão de abate'
    };
    return labels[field] || field;
  };

  // Função para submeter o formulário
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      // Preparar dados para envio
      const selectedHerd = herds.find(h => String(h.id) === String(formData.herdId));
      const cattleData = {
        name: formData.name,
        entryDate: formData.entryDate,
        origin: selectedHerd ? selectedHerd.name : null,
        earring: formData.name,
        herdId: formData.herdId || null,
        gender: formData.gender,
        breed: formData.breed,
        category: formData.category === 'outros' ? formData.customCategory : formData.category,
        entryWeight: parseFloat(formData.entryWeight),
        birthDate: formData.birthDate,
        age: calculateAge(formData.birthDate),
        targetWeight: parseFloat(formData.targetWeight),
        estimatedSlaughter: formData.estimatedSlaughter
      };

      // Chamar API para cadastrar gado
      await addCattle(cattleData);
      
      setSuccessMessage('Gado cadastrado com sucesso!');
      
      // Recarregar lista de gados
      await loadCattleList();
      
      // Limpar formulário
      setFormData({
        name: '',
        entryDate: '',
        
        herdId: '',
        gender: '',
        breed: '',
        category: '',
        customCategory: '',
        entryWeight: '',
        birthDate: '',
        targetWeight: '',
        estimatedSlaughter: ''
      });
      
      // Fechar formulário e mostrar lista
      setShowForm(false);

    } catch (error) {
      console.error('Erro ao cadastrar gado:', error);
      setErrorMessage(error.message || 'Erro ao cadastrar gado. Tente novamente.');
    } finally {
      setLoading(false);
    }
  };

  // Função para abrir formulário de cadastro
  const handleAddCattle = () => {
    setShowForm(true);
    setSuccessMessage('');
    setErrorMessage('');
  };

  // Função para cancelar cadastro
  const handleCancel = () => {
    setShowForm(false);
    setSuccessMessage('');
    setErrorMessage('');
    setFormData({
      name: '',
      entryDate: '',
      
      herdId: '',
      gender: '',
      breed: '',
      category: '',
      customCategory: '',
      entryWeight: '',
      birthDate: '',
      targetWeight: '',
      estimatedSlaughter: ''
    });
  };

  // Função para editar gado
  const handleEditCattle = (cattle) => {
    const herd = herds.find(h => String(h.id) === String(cattle.herd_id || cattle.herdId));
    const birthDate = cattle.birth_date || cattle.birthDate || '';
    const entryWeight = cattle.entry_weight ?? cattle.entryWeight ?? '';
    const targetWeight = cattle.target_weight ?? cattle.targetWeight ?? '';

    setSelectedCattle({
      ...cattle,
      herd_name: herd ? herd.name : cattle.origin,
      birth_date: birthDate ? birthDate.substring(0, 10) : '',
      entry_weight: entryWeight,
      target_weight: targetWeight
    });
    setShowEditModal(true);
  };

  // Função para abrir modal de confirmação de exclusão
  const handleDeleteCattle = (cattle) => {
    setCattleToDelete(cattle);
    setShowDeleteModal(true);
  };

  // Função para confirmar exclusão
  const confirmDeleteCattle = async () => {
    if (!cattleToDelete) return;
    
    try {
        await deleteCattle(cattleToDelete.id);
        setNotification({ visible: true, type: 'success', message: 'Gado excluído com sucesso!' });
        setTimeout(() => setNotification({ visible: false, type: 'success', message: '' }), 3500);
        setSuccessMessage('Gado excluído com sucesso!');
        setShowDeleteModal(false);
        setCattleToDelete(null);
        await loadCattleList();
    } catch (error) {
        console.error('Erro ao excluir gado:', error);
        const serverMessage = error?.message || error?.response?.data?.message || 'Erro ao excluir gado. Tente novamente.';
        setNotification({ visible: true, type: 'error', message: serverMessage });
        setTimeout(() => setNotification({ visible: false, type: 'error', message: '' }), 5000);
        setErrorMessage(serverMessage);
    }
  };

  // Função para cancelar exclusão
  const cancelDeleteCattle = () => {
    setShowDeleteModal(false);
    setCattleToDelete(null);
  };

  // Função para atualizar gado
  const handleUpdateCattle = async (e) => {
    e.preventDefault();
    
    try {
      // Build payload to match API v1 expected fields
      const selectedHerd = herds.find(h => String(h.id) === String(selectedCattle.herd_id || selectedCattle.herdId));
      const payload = {
        earring: selectedCattle.earring || selectedCattle.name,
        name: selectedCattle.name,
        breed: selectedCattle.breed,
        birth_date: selectedCattle.birth_date || selectedCattle.birthDate || null,
        // If user changed the herd, set origin to herd name so UI shows updated farm
        origin: selectedHerd ? selectedHerd.name : (selectedCattle.origin || null),
        gender: selectedCattle.gender || null,
        herd_id: selectedCattle.herd_id || selectedCattle.herdId || null,
        entry_weight: selectedCattle.entry_weight ?? selectedCattle.entryWeight ?? null,
        target_weight: selectedCattle.target_weight ?? selectedCattle.targetWeight ?? null
      };

      console.debug('DEBUG updateCattle payload ->', selectedCattle.id, payload);
      const resp = await updateCattle(selectedCattle.id, payload);
      // If backend returns a message, show it; otherwise show default success
      const successText = resp?.message || 'Gado atualizado com sucesso!';
      // Show toast notification (styled) instead of blocking alert
      setNotification({ visible: true, type: 'success', message: successText });
      setTimeout(() => setNotification({ visible: false, type: 'success', message: '' }), 3500);
      setSuccessMessage(successText);
      setShowEditModal(false);
      setSelectedCattle(null);
      await loadCattleList();
    } catch (error) {
      // Log detailed error for debugging and show server message when available
      console.error('Erro ao atualizar gado:', error);
      const serverMessage = error?.message || error?.response?.data?.message || (error?.response && JSON.stringify(error.response.data)) || null;
      setNotification({ visible: true, type: 'error', message: serverMessage || 'Erro ao atualizar gado. Tente novamente.' });
      setTimeout(() => setNotification({ visible: false, type: 'error', message: '' }), 5000);
      setErrorMessage(serverMessage || 'Erro ao atualizar gado. Tente novamente.');
    }
  };

  // Função para formatar data
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('pt-BR');
  };

  // Função para calcular idade
  const calculateAgeFromBirth = (birthDate) => {
    if (!birthDate) return 'N/A';
    const today = new Date();
    const birth = new Date(birthDate);
    const diffTime = Math.abs(today - birth);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 30) {
      return `${diffDays} dias`;
    } else if (diffDays < 365) {
      const months = Math.floor(diffDays / 30);
      return `${months} meses`;
    } else {
      const years = Math.floor(diffDays / 365);
      const months = Math.floor((diffDays % 365) / 30);
      return `${years} anos e ${months} meses`;
    }
  };

  return (
    <div className="cadastrar-gado-container">
      <Navbar />
      
      <div className="cadastrar-gado-content">
        <div className="cadastrar-gado-header">
          <h1>Gerenciar Gado</h1>
          <p>{showForm ? 'Preencha os dados do animal para cadastro no sistema' : 'Visualize e gerencie seus gados cadastrados'}</p>
        </div>

        {/* Lista de gados cadastrados */}
        {!showForm && (
          <div className="cattle-list-section">
            <div className="cattle-list-header">
              <h2>Gados Cadastrados ({cattleList.length})</h2>
              <button 
                className="add-cattle-btn"
                onClick={handleAddCattle}
              >
                + Adicionar Gado
              </button>
            </div>

            {listLoading ? (
              <div className="loading-message">
                <div className="loading-spinner"></div>
                <p>Carregando lista de gados...</p>
              </div>
            ) : cattleList.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">
                  <img src="/images/vaca.png" alt="Gado" />
                </div>
                <h3>Nenhum gado cadastrado</h3>
              </div>
            ) : (
              <div className="cattle-grid">
                {cattleList.map((cattle) => (
                  <div key={cattle.id} className="cattle-card">
                    <div className="cattle-card-header">
                      <h3>{cattle.name}</h3>
                      <span className={`status-badge ${cattle.status}`}>
                        {cattle.status}
                      </span>
                    </div>
                    <div className="cattle-card-body">
                      <div className="cattle-info">
                        <div className="info-item">
                          <span className="label">Raça:</span>
                          <span className="value">{cattle.breed}</span>
                        </div>
                        <div className="info-item">
                          <span className="label">Sexo:</span>
                          <span className="value">{cattle.gender}</span>
                        </div>
                              <div className="info-item">
                                <span className="label">Origem:</span>
                                <span className="value">{(() => {
                                  const herd = herds.find(h => String(h.id) === String(cattle.herd_id || cattle.herdId));
                                  return herd ? herd.name : (cattle.origin || '—');
                                })()}</span>
                              </div>
                        <div className="info-item">
                          <span className="label">Idade:</span>
                          <span className="value">{calculateAgeFromBirth(cattle.birth_date)}</span>
                        </div>
                        <div className="info-item">
                          <span className="label">Cadastrado em:</span>
                          <span className="value">{formatDate(cattle.created_at)}</span>
                        </div>
                      </div>
                    </div>
                    <div className="cattle-card-actions">
                      <button 
                        className="edit-cattle-btn"
                        onClick={() => handleEditCattle(cattle)}
                      >
                        Editar
                      </button>
           <button 
             className="delete-cattle-btn"
             onClick={() => handleDeleteCattle(cattle)}
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

        {/* Formulário de cadastro */}
        {showForm && (
          <form onSubmit={handleSubmit} className="cadastrar-gado-form">
          {/* Nome ou código interno */}
          <div className="form-group">
            <label htmlFor="name">Nome ou código interno do animal *</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              placeholder="Ex: GADO-001, Brinco 1234"
              required
            />
          </div>

          {/* Data de entrada */}
          <div className="form-group">
            <label htmlFor="entryDate">Data de entrada *</label>
            <input
              type="date"
              id="entryDate"
              name="entryDate"
              value={formData.entryDate}
              onChange={handleInputChange}
              required
            />
          </div>

          {/* (campo de procedência removido) */}

          {/* Seletor de Fazenda / Rebanho */}
          <div className="form-group">
            <label htmlFor="herdId">Fazenda / Rebanho *</label>
            <select
              id="herdId"
              name="herdId"
              value={formData.herdId}
              onChange={handleInputChange}
              required
            >
              <option value="">Selecione a fazenda</option>
              {herdsLoading ? (
                <option value="">Carregando...</option>
              ) : (
                herds.map(h => (
                  <option key={h.id} value={h.id}>{h.name}</option>
                ))
              )}
            </select>
          </div>

          {/* Sexo */}
          <div className="form-group">
            <label htmlFor="gender">Sexo *</label>
            <select
              id="gender"
              name="gender"
              value={formData.gender}
              onChange={handleInputChange}
              required
            >
              <option value="">Selecione o sexo</option>
              {genderOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Raça */}
          <div className="form-group">
            <label htmlFor="breed">Raça *</label>
            <input
              type="text"
              id="breed"
              name="breed"
              value={formData.breed}
              onChange={handleInputChange}
              placeholder="Ex: Nelore, Angus, Holandês"
              required
            />
          </div>

          {/* Categoria animal */}
          <div className="form-group">
            <label htmlFor="category">Categoria animal *</label>
            <select
              id="category"
              name="category"
              value={formData.category}
              onChange={handleInputChange}
              required
            >
              <option value="">Selecione a categoria</option>
              {categoryOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Campo para categoria customizada */}
          {formData.category === 'outros' && (
            <div className="form-group">
              <label htmlFor="customCategory">Especifique a categoria *</label>
              <input
                type="text"
                id="customCategory"
                name="customCategory"
                value={formData.customCategory}
                onChange={handleInputChange}
                placeholder="Digite a categoria personalizada"
                required
              />
            </div>
          )}

          {/* Peso de entrada */}
          <div className="form-group">
            <label htmlFor="entryWeight">Peso de entrada (kg) *</label>
            <input
              type="number"
              id="entryWeight"
              name="entryWeight"
              value={formData.entryWeight}
              onChange={handleInputChange}
              placeholder="Ex: 350"
              min="0"
              step="0.1"
              required
            />
          </div>

          {/* Data de nascimento */}
          <div className="form-group">
            <label htmlFor="birthDate">Data de nascimento *</label>
            <input
              type="date"
              id="birthDate"
              name="birthDate"
              value={formData.birthDate}
              onChange={handleInputChange}
              required
            />
            {formData.birthDate && (
              <div className="age-display">
                <small>Idade calculada: {calculateAge(formData.birthDate)}</small>
              </div>
            )}
          </div>

          {/* Peso de meta para saída */}
          <div className="form-group">
            <label htmlFor="targetWeight">Peso de meta para saída (kg) *</label>
            <input
              type="number"
              id="targetWeight"
              name="targetWeight"
              value={formData.targetWeight}
              onChange={handleInputChange}
              placeholder="Ex: 450"
              min="0"
              step="0.1"
              required
            />
          </div>

          {/* Previsão de abate */}
          <div className="form-group">
            <label htmlFor="estimatedSlaughter">Previsão de abate *</label>
            <input
              type="date"
              id="estimatedSlaughter"
              name="estimatedSlaughter"
              value={formData.estimatedSlaughter}
              onChange={handleInputChange}
              required
            />
          </div>

          {/* Mensagens de sucesso e erro */}
          {successMessage && (
            <div className="success-message">
              {successMessage}
            </div>
          )}

          {errorMessage && (
            <div className="error-message">
              {errorMessage}
            </div>
          )}

          {/* Botões de ação */}
          <div className="form-actions">
            <button type="button" className="cancel-btn" onClick={handleCancel}>
              Cancelar
            </button>
            <button 
              type="submit" 
              className="submit-btn"
              disabled={loading}
            >
              {loading ? 'Cadastrando...' : 'Cadastrar'}
            </button>
          </div>
        </form>
        )}

        {/* Modal de Edição */}
        {showEditModal && selectedCattle && (
          <div className="modal-overlay" onClick={() => setShowEditModal(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>Editar Gado</h3>
                <button className="close-btn" onClick={() => setShowEditModal(false)}>×</button>
              </div>
              
              <form onSubmit={handleUpdateCattle} className="edit-form">
                <div className="form-group">
                  <label>Nome *</label>
                  <input
                    type="text"
                    value={selectedCattle.name || ''}
                    onChange={(e) => setSelectedCattle({...selectedCattle, name: e.target.value})}
                    required
                  />
                </div>
                
                <div className="form-group">
                  <label>Raça *</label>
                  <input
                    type="text"
                    value={selectedCattle.breed || ''}
                    onChange={(e) => setSelectedCattle({...selectedCattle, breed: e.target.value})}
                    required
                  />
                </div>
                
                <div className="form-group">
                  <label>Sexo *</label>
                  <select
                    value={selectedCattle.gender || ''}
                    onChange={(e) => setSelectedCattle({...selectedCattle, gender: e.target.value})}
                    required
                  >
                    <option value="">Selecione o sexo</option>
                    <option value="macho">Macho</option>
                    <option value="femea">Fêmea</option>
                    <option value="castrado">Castrado</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label>Fazenda / Rebanho</label>
                  <select
                    value={selectedCattle.herd_id || selectedCattle.herdId || ''}
                    onChange={(e) => setSelectedCattle({...selectedCattle, herd_id: e.target.value ? Number(e.target.value) : null})}
                  >
                    <option value="">Selecione a fazenda</option>
                    {herds.map(h => (
                      <option key={h.id} value={h.id}>{h.name}</option>
                    ))}
                  </select>
                </div>
                
                <div className="form-group">
                  <label>Data de Nascimento</label>
                  <input
                    type="date"
                    value={selectedCattle.birth_date || ''}
                    onChange={(e) => setSelectedCattle({...selectedCattle, birth_date: e.target.value})}
                  />
                </div>

                <div className="form-group">
                  <label>Peso de Entrada (kg)</label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={selectedCattle.entry_weight ?? ''}
                    onChange={(e) => setSelectedCattle({
                      ...selectedCattle,
                      entry_weight: e.target.value !== '' ? Number(e.target.value) : ''
                    })}
                  />
                </div>

                <div className="form-group">
                  <label>Peso de Meta para Saída (kg)</label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={selectedCattle.target_weight ?? ''}
                    onChange={(e) => setSelectedCattle({
                      ...selectedCattle,
                      target_weight: e.target.value !== '' ? Number(e.target.value) : ''
                    })}
                  />
                </div>
                
                <div className="form-actions">
                  <button type="button" onClick={() => setShowEditModal(false)} className="cancel-btn">
                    Cancelar
                  </button>
                  <button type="submit" className="save-btn">
                    Salvar Alterações
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Modal de Confirmação de Exclusão */}
        {showDeleteModal && cattleToDelete && (
          <div className="modal-overlay" onClick={cancelDeleteCattle}>
            <div className="modal-content delete-confirmation-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div className="delete-icon">
                  ⚠️
                </div>
                <h3>Confirmar Exclusão</h3>
              </div>
              
              <div className="delete-message">
                <p>Tem certeza que deseja excluir o gado <strong>"{cattleToDelete.name}"</strong>?</p>
                <p className="warning-text">Esta ação não pode ser desfeita e todos os dados relacionados serão permanentemente removidos.</p>
              </div>
              
              <div className="form-actions">
                <button 
                  type="button" 
                  className="cancel-btn"
                  onClick={cancelDeleteCattle}
                >
                  Cancelar
                </button>
                <button 
                  type="button" 
                  className="delete-confirm-btn"
                  onClick={confirmDeleteCattle}
                >
                  Excluir Permanentemente
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      {/* Toast notification */}
      {notification.visible && (
        <div style={{
          position: 'fixed',
          right: 24,
          top: 24,
          zIndex: 9999,
          minWidth: 320,
          boxShadow: '0 6px 18px rgba(0,0,0,0.12)',
          borderRadius: 12,
          padding: '12px 16px',
          display: 'flex',
          alignItems: 'center',
          background: notification.type === 'success' ? 'linear-gradient(90deg,#06b6d4,#059669)' : 'linear-gradient(90deg,#ef4444,#9b2c2c)',
          color: 'white',
          fontWeight: 600
        }}>
          <div style={{marginRight: 12}}>{notification.type === 'success' ? '✓' : '⚠'}</div>
          <div style={{flex: 1, fontSize: 14}}>{notification.message}</div>
          <button onClick={() => setNotification({ visible: false, type: 'success', message: '' })} style={{
            background: 'transparent',
            border: 'none',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: 700,
            marginLeft: 12,
            cursor: 'pointer'
          }}>✕</button>
        </div>
      )}
    </div>
  );
};

export default CadastrarGado;
