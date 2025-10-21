import React, { useEffect, useMemo, useState } from 'react';
import { getCattle, addWeight, getWeightHistory, getWeightReport, getHerds, filterCattle } from '../../services/api';
import './WeightTracking.css';

const initialFilters = {
  startDate: '2025-08-01',
  endDate: '2025-08-01',
  minWeight: 200,
  maxWeight: 700,
  breeds: {
    nelore: false,
    girolando: false,
    angus: false,
    outro: false
  },
  situations: {
    aboveAverage: false,
    belowAverage: false,
    rapidGrowth: false,
    stable: false
  }
};

const WeightTracking = () => {
  const [herds, setHerds] = useState([]);
  const [herdsLoading, setHerdsLoading] = useState(true);
  const [herdError, setHerdError] = useState('');
  const [selectedHerd, setSelectedHerd] = useState(null);

  const [loading, setLoading] = useState(false);
  const [allCattleData, setAllCattleData] = useState([]);
  const [cattleData, setCattleData] = useState([]);
  const [mainCattle, setMainCattle] = useState(null);
  const [weightStats, setWeightStats] = useState({ averageWeight: null, maxWeight: null, minWeight: null });

  const [showAddWeightForm, setShowAddWeightForm] = useState(false);
  const [showFilterModal, setShowFilterModal] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const [newWeight, setNewWeight] = useState({
    cattleId: '',
    weight: '',
    date: new Date().toISOString().split('T')[0]
  });

  const [filters, setFilters] = useState(initialFilters);

  const hasCattleData = useMemo(() => cattleData.length > 0, [cattleData]);

  useEffect(() => {
    const fetchHerds = async () => {
      setHerdsLoading(true);
      setHerdError('');
      try {
        const response = await getHerds();
        const list = Array.isArray(response) ? response : (response?.herds || []);
        setHerds(list);
      } catch (error) {
        console.error('Erro ao carregar fazendas:', error);
        setHerdError('Não foi possível carregar as fazendas. Tente novamente.');
      } finally {
        setHerdsLoading(false);
      }
    };

    fetchHerds();
  }, []);

  useEffect(() => {
    if (!selectedHerd) return;
    loadCattle(selectedHerd.id);
  }, [selectedHerd]);

  useEffect(() => {
    if (hasCattleData) {
      setNewWeight(prev => ({ ...prev, cattleId: prev.cattleId || cattleData[0].id }));
    } else {
      setNewWeight(prev => ({ ...prev, cattleId: '' }));
    }
  }, [hasCattleData, cattleData]);

  const toNumber = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const formatDateLabel = (dateString) => {
    if (!dateString) return '—';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) {
      return dateString;
    }
    return date.toLocaleDateString('pt-BR');
  };

  const createFallbackHistory = (weight) => {
    const numericWeight = toNumber(weight) ?? 0;
    return [
      {
        date: new Date().toLocaleDateString('pt-BR'),
        weight: Number(numericWeight.toFixed(1))
      }
    ];
  };

  const normalizeHistory = (rawHistory, fallbackWeight) => {
    if (!Array.isArray(rawHistory)) {
      return createFallbackHistory(fallbackWeight);
    }

    const formatted = rawHistory
      .map((item) => {
        const weight = toNumber(item.weight);
        if (weight === null) return null;
        return {
          date: formatDateLabel(item.date),
          weight: Number(weight.toFixed(1))
        };
      })
      .filter(Boolean);

    if (formatted.length > 0) {
      return formatted;
    }

    return createFallbackHistory(fallbackWeight);
  };

  const updateStats = (data) => {
    if (!Array.isArray(data) || data.length === 0) {
      setWeightStats({ averageWeight: null, maxWeight: null, minWeight: null });
      return;
    }

    const weights = data
      .map((item) => toNumber(item.currentWeight))
      .filter((value) => value !== null);

    if (weights.length === 0) {
      setWeightStats({ averageWeight: null, maxWeight: null, minWeight: null });
      return;
    }

    const total = weights.reduce((acc, weight) => acc + weight, 0);
    setWeightStats({
      averageWeight: Number((total / weights.length).toFixed(1)),
      maxWeight: Math.max(...weights),
      minWeight: Math.min(...weights)
    });
  };

  const fetchAnimalData = async (animals) => {
    return Promise.all(
      animals.map(async (animal) => {
        try {
          const historyResponse = await getWeightHistory(animal.id);
          const rawHistory = historyResponse?.history ?? historyResponse ?? [];
          const fallbackWeight =
            toNumber(rawHistory?.[0]?.weight) ??
            toNumber(animal.latest_weight) ??
            toNumber(animal.entry_weight) ??
            toNumber(animal.currentWeight) ??
            0;

          const history = normalizeHistory(rawHistory, fallbackWeight);
          const currentWeight = history[0]?.weight ?? fallbackWeight ?? 0;

          return {
            id: animal.id,
            name: animal.name || animal.earring || `Gado ${animal.id}`,
            currentWeight,
            weightHistory: history,
            chartData: history,
            meta: {
              breed: animal.breed || '',
              gender: animal.gender || '',
              status: animal.status || ''
            }
          };
        } catch (error) {
          console.error(`Erro ao carregar histórico do gado ${animal.name || animal.earring}:`, error);
          const fallbackHistory = createFallbackHistory(animal.entry_weight);
          return {
            id: animal.id,
            name: animal.name || animal.earring || `Gado ${animal.id}`,
            currentWeight: fallbackHistory[0].weight,
            weightHistory: fallbackHistory,
            chartData: fallbackHistory,
            meta: {
              breed: animal.breed || '',
              gender: animal.gender || '',
              status: animal.status || ''
            }
          };
        }
      })
    );
  };

  const loadCattle = async (herdId) => {
    setLoading(true);
    try {
      const response = await getCattle({ herdId });
      const animals = Array.isArray(response)
        ? response
        : response?.animals || response?.cattle || [];

      const processed = animals.length > 0 ? await fetchAnimalData(animals) : [];

      setAllCattleData(processed);
      setCattleData(processed);
      setMainCattle(processed[0] || null);
      updateStats(processed);
    } catch (error) {
      console.error('Erro ao carregar bovinos:', error);
      setAllCattleData([]);
      setCattleData([]);
      setMainCattle(null);
      updateStats([]);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    if (!selectedHerd) return;

    setIsRefreshing(true);
    try {
      await loadCattle(selectedHerd.id);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleOpenReport = async () => {
    setIsRefreshing(true);
    try {
      await getWeightReport();
      window.location.href = '/relatorio-gados';
    } catch (error) {
      console.error('Erro ao gerar relatório:', error);
      alert('Não foi possível gerar o relatório agora. Tente novamente em instantes.');
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleAddWeight = async (e) => {
    e.preventDefault();

    if (!newWeight.cattleId) {
      alert('Selecione um gado para registrar o peso.');
      return;
    }

    try {
      await addWeight(newWeight);

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
      successMessage.innerHTML = '✓ Peso adicionado com sucesso!';
      document.body.appendChild(successMessage);

      setTimeout(() => {
        successMessage.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => document.body.removeChild(successMessage), 300);
      }, 3000);

      setNewWeight({
        cattleId: '',
        weight: '',
        date: new Date().toISOString().split('T')[0]
      });
      setShowAddWeightForm(false);

      if (selectedHerd) {
        await loadCattle(selectedHerd.id);
      }
    } catch (error) {
      console.error('Erro ao adicionar peso:', error);
      alert('Erro ao adicionar peso. Tente novamente.');
    }
  };

  const handleFilterChange = (section, key, value) => {
    setFilters((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  const handleNumericFilterChange = (key, value) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value
    }));
  };

  const applyFiltersToData = () => {
    let filtered = [...allCattleData];

    const min = toNumber(filters.minWeight);
    const max = toNumber(filters.maxWeight);

    if (min !== null) {
      filtered = filtered.filter((item) => toNumber(item.currentWeight) >= min);
    }

    if (max !== null) {
      filtered = filtered.filter((item) => toNumber(item.currentWeight) <= max);
    }

    const activeBreeds = Object.entries(filters.breeds)
      .filter(([, value]) => value)
      .map(([key]) => key.toLowerCase());

    if (activeBreeds.length > 0) {
      filtered = filtered.filter((item) => {
        const breed = (item.meta?.breed || '').toLowerCase();
        return activeBreeds.some((active) => breed.includes(active));
      });
    }

    return filtered;
  };

  const handleApplyFilters = async () => {
    try {
      setIsRefreshing(true);
      const payload = {
        herdId: selectedHerd?.id,
        minWeight: filters.minWeight,
        maxWeight: filters.maxWeight,
        breeds: filters.breeds,
        situations: filters.situations,
        startDate: filters.startDate,
        endDate: filters.endDate
      };
      const response = await filterCattle(payload);
      const list = response?.cattle || [];
      const processed = list.length > 0 ? await fetchAnimalData(list) : [];
      setCattleData(processed);
      updateStats(processed);
      setMainCattle(processed[0] || null);
    } catch (error) {
      console.error('Erro ao aplicar filtros:', error);
    } finally {
      setIsRefreshing(false);
      setShowFilterModal(false);
    }
  };

  const handleClearFilters = () => {
    setFilters(initialFilters);
    setCattleData(allCattleData);
    setMainCattle(allCattleData[0] || null);
    updateStats(allCattleData);
  };

  const formatSituationLabel = (key) => {
    switch (key) {
      case 'aboveAverage':
        return 'Acima da média';
      case 'belowAverage':
        return 'Abaixo da média';
      case 'rapidGrowth':
        return 'Crescimento rápido';
      case 'stable':
        return 'Estável';
      default:
        return key;
    }
  };

  const renderStatValue = (value) => (value !== null ? `${value} Kg` : '—');

  const handleSelectCattle = (selectedCattle) => {
    setMainCattle({
      id: selectedCattle.id,
      name: selectedCattle.name,
      currentWeight: selectedCattle.currentWeight,
      weightHistory: selectedCattle.weightHistory,
      chartData: selectedCattle.weightHistory
    });
  };

  const handleChangeHerd = () => {
    setSelectedHerd(null);
    setAllCattleData([]);
    setCattleData([]);
    setMainCattle(null);
    setWeightStats({ averageWeight: null, maxWeight: null, minWeight: null });
    setFilters(initialFilters);
  };

  const renderHerdSelection = () => (
    <div className="herd-selection-page">
      <div className="weight-selection-wrapper">
        <div className="weight-selection-content">
          <h1 className="weight-selection-title">Selecione uma fazenda</h1>
          <p className="weight-selection-subtitle">
            Escolha qual rebanho deseja acompanhar. Você pode trocar de fazenda quando quiser.
          </p>

          {herdsLoading ? (
            <div className="weight-selection-loading">
              <div className="loading-spinner"></div>
              <span>Carregando fazendas...</span>
            </div>
          ) : herdError ? (
            <div className="weight-selection-error">{herdError}</div>
          ) : herds.length === 0 ? (
            <div className="weight-selection-empty">
              <p>Nenhuma fazenda cadastrada até o momento.</p>
              <a href="/criar-fazenda" className="weight-selection-link">Ir para Criar Fazenda</a>
            </div>
          ) : (
            <div className="weight-selection-grid">
              {herds.map((herd) => (
                <button
                  key={herd.id}
                  className="herd-card"
                  onClick={() => setSelectedHerd(herd)}
                >
                  <div className="herd-card-title">{herd.name}</div>
                  {herd.location && <div className="herd-card-info">{herd.location}</div>}
                  {herd.city && <div className="herd-card-info">{herd.city}</div>}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  if (!selectedHerd) {
    return (
      <div className="herd-selection-page">
        <div className="weight-selection-wrapper">
          <div className="weight-selection-content">
            <h1 className="weight-selection-title">Selecione uma fazenda</h1>
            <p className="weight-selection-subtitle">
              Escolha qual rebanho deseja acompanhar. Você pode trocar de fazenda quando quiser.
            </p>

            {herdsLoading ? (
              <div className="weight-selection-loading">
                <div className="loading-spinner"></div>
                <span>Carregando fazendas...</span>
              </div>
            ) : herdError ? (
              <div className="weight-selection-error">{herdError}</div>
            ) : herds.length === 0 ? (
              <div className="weight-selection-empty">
                <p>Nenhuma fazenda cadastrada até o momento.</p>
                <a href="/criar-fazenda" className="weight-selection-link">Ir para Criar Fazenda</a>
              </div>
            ) : (
              <div className="weight-selection-grid">
                {herds.map((herd) => (
                  <button
                    key={herd.id}
                    className="herd-card"
                    onClick={() => setSelectedHerd(herd)}
                  >
                    <div className="herd-card-title">{herd.name}</div>
                    {herd.location && <div className="herd-card-info">{herd.location}</div>}
                    {herd.city && <div className="herd-card-info">{herd.city}</div>}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="weight-tracking-container">
      <div className="weight-tracking-content">
        <div className="weight-tracking-header">
          <div className="weight-tracking-title-section">
            <div className="weight-title-block">
              <h1 className="weight-tracking-title">Acompanhamento de Peso</h1>
              <div className="selected-herd-indicator">
                <span>
                  Fazenda selecionada: <strong>{selectedHerd?.name}</strong>
                </span>
                <button className="change-herd-button" onClick={handleChangeHerd}>
                  Trocar fazenda
                </button>
              </div>
            </div>
            <div className="weight-tracking-actions">
              <button
                className="weight-tracking-button add-weight"
                onClick={() => setShowAddWeightForm(true)}
                disabled={!hasCattleData}
              >
                <div className="weight-tracking-button-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
                  </svg>
                </div>
                Adicionar Peso
              </button>
              <button
                className="weight-tracking-button filter-cattle"
                onClick={() => setShowFilterModal(true)}
                disabled={!hasCattleData}
              >
                <div className="weight-tracking-button-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z" />
                  </svg>
                </div>
                Filtrar Gado
              </button>
              <button
                className={`weight-tracking-button refresh ${isRefreshing ? 'refreshing' : ''}`}
                onClick={handleRefresh}
                disabled={isRefreshing || !selectedHerd}
              >
                <div className="weight-tracking-button-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
                  </svg>
                </div>
                {isRefreshing ? 'Atualizando...' : 'Atualizar'}
              </button>
              <button
                className="weight-tracking-button report"
                onClick={handleOpenReport}
                disabled={isRefreshing}
              >
                <div className="weight-tracking-button-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 5h2l0.4 2M7 5h14M9 9h12l-1 9H6L5 5H3" />
                  </svg>
                </div>
                Relatório automático
              </button>
            </div>
          </div>
        </div>

        <div className="weight-tracking-stats">
          <div className="weight-stats-grid">
            <div className="weight-stat-card">
              <div className="weight-stat-title">Peso Médio do Rebanho</div>
              <div className="weight-stat-value">{renderStatValue(weightStats.averageWeight)}</div>
            </div>
            <div className="weight-stat-card">
              <div className="weight-stat-title">Maior Peso</div>
              <div className="weight-stat-value">{renderStatValue(weightStats.maxWeight)}</div>
            </div>
            <div className="weight-stat-card">
              <div className="weight-stat-title">Menor Peso</div>
              <div className="weight-stat-value">{renderStatValue(weightStats.minWeight)}</div>
            </div>
          </div>
        </div>

        <div className="weight-tracking-main">
          {!hasCattleData ? (
            <div className="weight-empty-state">
              <h3>Nenhum gado encontrado nesta fazenda</h3>
              <p>Cadastre um gado ou registre uma pesagem recente para visualizar o acompanhamento.</p>
            </div>
          ) : (
            <div className="weight-tracking-layout">
              <div className="weight-main-cattle-section">
                {mainCattle ? (
                  <div className="weight-main-cattle-card">
                    <div className="weight-main-cattle-header">
                      <div className="weight-main-cattle-icon">
                        <img src="/images/vaca.png" alt="Vaca" style={{ width: '24px', height: '24px' }} />
                      </div>
                      <div className="weight-main-cattle-name">{mainCattle.name}</div>
                    </div>
                    <div className="weight-main-cattle-current">
                      Peso Atual: <span className="weight-main-cattle-weight">{mainCattle.currentWeight} Kg</span>
                    </div>
                    <div className="weight-main-cattle-history">
                      <div className="weight-main-cattle-history-title">HISTÓRICO DE PESO</div>
                      <div className="weight-main-cattle-history-list">
                        {mainCattle.weightHistory.map((entry, index) => (
                          <div key={index} className="weight-main-cattle-history-item">
                            <span>{entry.date}</span>
                            <span>{entry.weight} Kg</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="weight-line-chart-section">
                      <div className="weight-line-chart-header">
                        <h4 className="weight-line-chart-title">Gráfico de Peso</h4>
                      </div>
                      <div className="weight-line-chart-container">
                        <div className="weight-line-chart">
                          <div className="weight-line-chart-y-axis">
                            {(() => {
                              const weights = mainCattle.chartData.map((p) => p.weight);
                              const minWeight = Math.min(...weights);
                              const maxWeight = Math.max(...weights);
                              const range = maxWeight - minWeight;
                              const step = range / 4 || 1;

                              return [4, 3, 2, 1, 0].map((i) => {
                                const value = Number((maxWeight - step * i).toFixed(1));
                                return (
                                  <div key={i} className="weight-line-chart-scale">
                                    {Number.isFinite(value) ? value : 0}
                                  </div>
                                );
                              });
                            })()}
                          </div>
                          <div className="weight-line-chart-content">
                            <svg className="weight-line-chart-svg" viewBox="0 0 400 200">
                              <defs>
                                <pattern id="grid" width="80" height="40" patternUnits="userSpaceOnUse">
                                  <path d="M 80 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="0.5" />
                                </pattern>
                              </defs>
                              <rect width="100%" height="100%" fill="url(#grid)" />

                              <polyline
                                points={mainCattle.chartData
                                  .map((point, index) => {
                                    const x = index * 80 + 40;
                                    const weights = mainCattle.chartData.map((p) => p.weight);
                                    const minWeight = Math.min(...weights);
                                    const maxWeight = Math.max(...weights);
                                    const weightRange = maxWeight - minWeight;
                                    const y = weightRange > 0 ? 180 - ((point.weight - minWeight) / weightRange) * 160 : 90;
                                    return `${x},${y}`;
                                  })
                                  .join(' ')}
                                fill="none"
                                stroke="#15AABF"
                                strokeWidth="3"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />

                              {mainCattle.chartData.map((point, index) => {
                                const x = index * 80 + 40;
                                const weights = mainCattle.chartData.map((p) => p.weight);
                                const minWeight = Math.min(...weights);
                                const maxWeight = Math.max(...weights);
                                const weightRange = maxWeight - minWeight;
                                const y = weightRange > 0 ? 180 - ((point.weight - minWeight) / weightRange) * 160 : 90;

                                return (
                                  <g key={index}>
                                    <circle cx={x} cy={y} r="6" fill="white" stroke="#15AABF" strokeWidth="2" />
                                    <circle cx={x} cy={y} r="3" fill="#15AABF" />
                                    <text
                                      x={x}
                                      y={y - 12}
                                      textAnchor="middle"
                                      fontSize="10"
                                      fill="#2c3e50"
                                      fontWeight="600"
                                    >
                                      {point.weight}kg
                                    </text>
                                  </g>
                                );
                              })}
                            </svg>
                            <div className="weight-line-chart-x-labels">
                              {mainCattle.chartData.map((point, index) => (
                                <div key={index} className="weight-line-chart-x-label">
                                  {point.date}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="weight-main-placeholder">
                    <p>Selecione um gado na lista ao lado para visualizar o acompanhamento detalhado.</p>
                  </div>
                )}
              </div>

              <div className="weight-other-cattle-section">
                {cattleData.map((cattle) => (
                  <div
                    key={cattle.id}
                    className={`weight-other-cattle-card ${mainCattle?.id === cattle.id ? 'selected' : ''}`}
                    onClick={() => handleSelectCattle(cattle)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="weight-other-cattle-header">
                      <div className="weight-other-cattle-icon">
                        <img src="/images/vaca.png" alt="Vaca" />
                      </div>
                      <div className="weight-other-cattle-name">{cattle.name}</div>
                    </div>
                    <div className="weight-other-cattle-current">
                      Peso Atual: {cattle.currentWeight} Kg
                    </div>
                    <div className="weight-other-cattle-history">
                      <div className="weight-other-cattle-history-title">Histórico de Peso</div>
                      <div className="weight-other-cattle-history-list">
                        {cattle.weightHistory.map((entry, index) => (
                          <div key={index} className="weight-other-cattle-history-item">
                            <span>{entry.date}</span>
                            <span>{entry.weight} Kg</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {showAddWeightForm && (
        <div className="weight-tracking-modal">
          <div className="weight-tracking-modal-content">
            <h3 className="weight-tracking-modal-title">Adicionar Peso</h3>
            <form onSubmit={handleAddWeight} className="weight-tracking-form">
              <div className="weight-tracking-form-group">
                <label className="weight-tracking-form-label">Selecionar Gado</label>
                <select
                  value={newWeight.cattleId}
                  onChange={(e) => setNewWeight({ ...newWeight, cattleId: e.target.value })}
                  className="weight-tracking-form-input"
                  required
                >
                  <option value="">Selecione um gado</option>
                  {cattleData.map((cattle) => (
                    <option key={cattle.id} value={cattle.id}>
                      {cattle.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="weight-tracking-form-group">
                <label className="weight-tracking-form-label">Peso (Kg)</label>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={newWeight.weight}
                  onChange={(e) => setNewWeight({ ...newWeight, weight: e.target.value })}
                  className="weight-tracking-form-input"
                  required
                />
              </div>
              <div className="weight-tracking-form-group">
                <label className="weight-tracking-form-label">Data</label>
                <input
                  type="date"
                  value={newWeight.date}
                  onChange={(e) => setNewWeight({ ...newWeight, date: e.target.value })}
                  className="weight-tracking-form-input"
                  required
                />
              </div>
              <div className="weight-tracking-form-actions">
                <button
                  type="button"
                  onClick={() => setShowAddWeightForm(false)}
                  className="weight-tracking-form-button cancel"
                >
                  Cancelar
                </button>
                <button type="submit" className="weight-tracking-form-button submit">
                  Adicionar
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showFilterModal && (
        <div className="weight-tracking-modal">
          <div className="weight-tracking-filter-modal">
            <h3 className="weight-tracking-modal-title">Filtrar Gado</h3>

            <div className="weight-filter-content">
              <div className="weight-filter-section">
                <label className="weight-filter-label">Período</label>
                <div className="weight-filter-date-range">
                  <input
                    type="date"
                    value={filters.startDate}
                    onChange={(e) => handleNumericFilterChange('startDate', e.target.value)}
                    className="weight-filter-date-input"
                  />
                  <span className="weight-filter-date-separator">—</span>
                  <input
                    type="date"
                    value={filters.endDate}
                    onChange={(e) => handleNumericFilterChange('endDate', e.target.value)}
                    className="weight-filter-date-input"
                  />
                </div>
              </div>

              <div className="weight-filter-section">
                <label className="weight-filter-label">Peso (Kg)</label>
                <div className="weight-filter-range-wrapper">
                  <div className="weight-filter-slider-container">
                    <div className="weight-filter-slider-value">
                      <span className="weight-filter-current-value">{filters.minWeight} Kg</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1000"
                      value={filters.minWeight}
                      onChange={(e) => handleNumericFilterChange('minWeight', Number(e.target.value))}
                      className="weight-filter-slider"
                    />
                    <div className="weight-filter-slider-labels">
                      <span>0 Kg</span>
                      <span>1000 Kg</span>
                    </div>
                  </div>
                  <div className="weight-filter-slider-container">
                    <div className="weight-filter-slider-value">
                      <span className="weight-filter-current-value">{filters.maxWeight} Kg</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1000"
                      value={filters.maxWeight}
                      onChange={(e) => handleNumericFilterChange('maxWeight', Number(e.target.value))}
                      className="weight-filter-slider"
                    />
                    <div className="weight-filter-slider-labels">
                      <span>0 Kg</span>
                      <span>1000 Kg</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="weight-filter-section">
                <label className="weight-filter-label">Raça</label>
                <div className="weight-filter-checkboxes">
                  {Object.keys(filters.breeds).map((breedKey) => (
                    <label className="weight-filter-checkbox" key={breedKey}>
                      <input
                        type="checkbox"
                        checked={filters.breeds[breedKey]}
                        onChange={(e) => handleFilterChange('breeds', breedKey, e.target.checked)}
                      />
                      <span className="weight-filter-checkmark"></span>
                      {breedKey.charAt(0).toUpperCase() + breedKey.slice(1)}
                    </label>
                  ))}
                </div>
              </div>

              <div className="weight-filter-section">
                <label className="weight-filter-label">Situação</label>
                <div className="weight-filter-checkboxes">
                  {Object.entries(filters.situations).map(([key, value]) => (
                    <label className="weight-filter-checkbox" key={key}>
                      <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) => handleFilterChange('situations', key, e.target.checked)}
                      />
                      <span className="weight-filter-checkmark"></span>
                      {formatSituationLabel(key)}
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="weight-filter-actions">
              <button onClick={handleClearFilters} className="weight-filter-button clear">
                Limpar Filtros
              </button>
              <button onClick={handleApplyFilters} className="weight-filter-button apply">
                Aplicar Filtros
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WeightTracking;
