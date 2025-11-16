import React, { useState, useEffect, useMemo, useRef } from 'react';
import { FaTractor, FaWeightHanging, FaBell, FaCheck } from 'react-icons/fa';
import { useAuth } from '../../contexts/AuthContext';
import { getCattle, addCattle, updateCattle, deleteCattle, getDashboardData, getRecentActivities, getCurrentUser } from '../../services/api';
import './Dashboard.css';

const Dashboard = () => {
  const { user, userId } = useAuth();
  const [cattle, setCattle] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [showNotifications, setShowNotifications] = useState(false);
  const [readNotifications, setReadNotifications] = useState([]);
  const [newCattle, setNewCattle] = useState({
    name: '',
    breed: '',
    age: '',
    weight: '',
    healthStatus: 'healthy'
  });

  // Estados para dados do dashboard
  const [dashboardData, setDashboardData] = useState({
    totalCattle: 0,
    excellentCattle: 0,
    reasonableCattle: 0,
    alertCattle: 0,
    lastRegistration: '',
    activeFarm: ''
  });

  const [recentActivities, setRecentActivities] = useState([]);

  const [lastUpdated, setLastUpdated] = useState(null);
  const notificationsRef = useRef(null);

  useEffect(() => {
    // Initial load
    loadDashboardData();

    // Listen for global data-change events dispatched by api service
    const handler = () => {
      // Recarregar dados quando alguma mutação ocorrer (add/update/delete)
      loadDashboardData();
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('bovicare:data-changed', handler);
    }

    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('bovicare:data-changed', handler);
      }
    };
  }, []);

  const buildDescriptionFromActivity = (activity) => {
    const name = activity.object_name || activity.object_id || '';
    switch (activity.object_type) {
      case 'animal':
        return `Novo gado cadastrado: ${name}`;
      case 'herd':
        return `Fazenda cadastrada: ${name}`;
      case 'weighing':
        return `Peso de ${activity.extra?.weight || activity.weight || ''} Kg registrado`;
      default:
        return activity.description || 'Atividade registrada';
    }
  };

  const deriveActivityIcon = (activity) => {
    if (activity.icon && !['Cadastro', 'cadastrargado', 'vaca'].includes(activity.icon)) {
      return activity.icon;
    }

    switch (activity.object_type) {
      case 'animal':
        return 'vaca';
      case 'herd':
        return 'herd';
      case 'weighing':
        return 'scale';
      default:
        return activity.icon || 'vaca';
    }
  };

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Carregar dados do dashboard em paralelo
      const storedUser = (() => {
        try {
          return JSON.parse(localStorage.getItem('user') || '{}');
        } catch (e) {
          return {};
        }
      })();

      const effectiveUserId = userId || storedUser.id;
      const effectiveUsername = user?.username || storedUser.username;

      const [cattleData, dashboardDataResult, activitiesData, userData] = await Promise.allSettled([
        getCattle(),
        getDashboardData(),
        getRecentActivities({ user_id: effectiveUserId, username: effectiveUsername }),
        getCurrentUser(userId)
      ]);

      // Debug: log statuses for inspection
      if (cattleData.status !== 'fulfilled') console.error('getCattle failed:', cattleData.reason);
      if (dashboardDataResult.status !== 'fulfilled') console.error('getDashboardData failed:', dashboardDataResult.reason);
      if (activitiesData.status !== 'fulfilled') console.error('getRecentActivities failed:', activitiesData.reason);
      if (userData.status !== 'fulfilled') console.error('getCurrentUser failed:', userData.reason);

      // Atualizar dados de bovinos
      let cattleList = [];
      if (cattleData.status === 'fulfilled' && Array.isArray(cattleData.value)) {
        cattleList = cattleData.value;
        setCattle(cattleList);
      }

      // Use dashboard endpoint result directly if available
      if (dashboardDataResult.status === 'fulfilled' && dashboardDataResult.value) {
        const d = dashboardDataResult.value;
        let defaultFarmName = '';
        if (d.herds_distribution && d.herds_distribution.length > 0) {
          const firstWithName = d.herds_distribution.find((item) => item && item.name);
          defaultFarmName = firstWithName?.name || '';
        }
        setDashboardData({
          totalCattle: d.total_animals || 0,
          excellentCattle: d.active_animals || 0,
          reasonableCattle: 0,
          alertCattle: 0,
          lastRegistration: d.recent_weighings && d.recent_weighings.length > 0 ? d.recent_weighings[0].created_at || d.recent_weighings[0].date || '' : '',
          activeFarm: defaultFarmName
        });
        // If dashboard returns recent weighings, convert them to activity-like items
        const shouldMergeWeighings = !effectiveUserId && !effectiveUsername;

        if (shouldMergeWeighings && Array.isArray(d.recent_weighings) && d.recent_weighings.length > 0) {
          const weighActivities = d.recent_weighings.map((w, idx) => ({
            id: `weigh-${w.id || idx}`,
            type: 'weight',
            description: `Peso de ${w.weight} Kg adicionado para GADO ${w.animal_id || ''}`,
            date: w.date || w.created_at || '',
            icon: 'vaca'
          }));

          // Prepend weighActivities to activitiesData if activitiesData provided later
          if (activitiesData.status === 'fulfilled' && Array.isArray(activitiesData.value)) {
            setRecentActivities([
              ...weighActivities,
              ...activitiesData.value.map((a) => ({ ...a, date: a.created_at || a.date || a.datetime || '' }))
            ]);
          } else {
            setRecentActivities(weighActivities.map((w) => ({ ...w, date: w.date || w.created_at || '' })));
          }
        }
      } else {
        // dashboard endpoint not available; set conservative values
        setDashboardData((prev) => ({ ...prev, totalCattle: Array.isArray(cattleList) ? cattleList.length : 0 }));
      }

      // If getRecentActivities returned and hasn't been applied above, apply it and normalize dates
      if (activitiesData.status === 'fulfilled' && Array.isArray(activitiesData.value)) {
        const normalized = activitiesData.value.map((a) => ({
          ...a,
          date: a.created_at || a.date || a.datetime || '',
          icon: deriveActivityIcon(a),
          description: a.description || buildDescriptionFromActivity(a)
        }));

        // Merge with existing recentActivities (weighings may have been prepended earlier)
        setRecentActivities((prev) => {
          const existingIds = new Set(prev.map((p) => p.id));
          const merged = [...prev];
          normalized.forEach((it) => {
            if (!existingIds.has(it.id)) merged.push(it);
          });
          return merged.length ? merged : normalized;
        });
      } else {
        setRecentActivities([]);
      }

      // Atualizar dados do usuário
      if (userData.status === 'fulfilled') {
        setCurrentUser(userData.value);
      }

    } catch (error) {
      console.error('Erro ao carregar dados do dashboard:', error);
      // Manter dados padrão em caso de erro
    } finally {
      setLoading(false);
      setLastUpdated(new Date());
    }
  };

  const notificationItems = useMemo(() => {
    const normalizeDate = (dateStr) => {
      if (!dateStr) return null;
      const parsed = new Date(dateStr);
      return Number.isNaN(parsed.getTime()) ? null : parsed;
    };

    const mapActivityToNotification = (activity) => {
      const baseDate = normalizeDate(activity.created_at || activity.date || activity.datetime);
      let title = 'Atualização';
      let type = 'default';

      switch (activity.object_type) {
        case 'animal':
          title = activity.action === 'update' ? 'Animal atualizado' : 'Novo gado cadastrado';
          type = 'animal';
          break;
        case 'herd':
          title = activity.action === 'update' ? 'Fazenda atualizada' : 'Nova fazenda criada';
          type = 'farm';
          break;
        case 'weighing':
          title = 'Peso registrado';
          type = activity.action === 'alert' ? 'alert' : 'weight';
          break;
        default:
          if (/alerta/i.test(activity.description || '')) {
            title = 'Alerta do rebanho';
            type = 'alert';
          } else {
            title = activity.action === 'update' ? 'Atualização realizada' : 'Atividade registrada';
          }
      }

      return {
        id: activity.id || `${activity.object_type || 'activity'}-${activity.object_id || Math.random()}`,
        title,
        message: activity.description || buildDescriptionFromActivity(activity),
        date: baseDate,
        rawDate: activity.created_at || activity.date || activity.datetime || '',
        type
      };
    };

    if (!Array.isArray(recentActivities)) return [];
    return recentActivities.map(mapActivityToNotification);
  }, [recentActivities]);

  const unreadCount = useMemo(() => {
    const readSet = new Set(readNotifications);
    return notificationItems.filter(item => !readSet.has(item.id)).length;
  }, [notificationItems, readNotifications]);

  useEffect(() => {
    // Remove ids that no longer exist
    const ids = notificationItems.map(item => item.id);
    setReadNotifications(prev => prev.filter(id => ids.includes(id)));
  }, [notificationItems]);

  useEffect(() => {
    if (!showNotifications) return;

    const handleClickOutside = (event) => {
      if (notificationsRef.current && !notificationsRef.current.contains(event.target)) {
        setShowNotifications(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showNotifications]);

  const handleToggleNotifications = () => {
    setShowNotifications(prev => !prev);
  };

  const handleMarkAllRead = () => {
    setReadNotifications(notificationItems.map(item => item.id));
  };

  const loadCattle = async () => {
    try {
      setLoading(true);
      const data = await getCattle();
      setCattle(data);
    } catch (error) {
      console.error('Erro ao carregar bovinos:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddCattle = async (e) => {
    e.preventDefault();
    try {
      await addCattle(newCattle);
      setNewCattle({
        name: '',
        breed: '',
        age: '',
        weight: '',
        healthStatus: 'healthy'
      });
      setShowAddForm(false);
      loadCattle();
    } catch (error) {
      console.error('Erro ao adicionar bovino:', error);
    }
  };

  const handleDeleteCattle = async (id) => {
    if (window.confirm('Tem certeza que deseja deletar este bovino?')) {
      try {
        await deleteCattle(id);
        loadCattle();
      } catch (error) {
        console.error('Erro ao deletar bovino:', error);
      }
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Filtrar atividades relacionadas a leite/produção
  const filteredActivities = recentActivities.filter(
    (a) => a.type !== 'production' && !/leite/i.test(a.description || '')
  );

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <div className="dashboard-content">
          <div className="dashboard-title-section">
            <div>
              <h1 className="dashboard-title">
                Bom dia, {currentUser?.username || user?.username || 'Caio'}!
              </h1>
              <p className="dashboard-subtitle">
                {currentUser?.farm_name || dashboardData.activeFarm || ''}
              </p>
              {lastUpdated && (
                <p className="dashboard-last-updated" style={{fontSize: '12px', color: '#6b7280', marginTop: 6}}>
                  Última atualização: {lastUpdated.toLocaleTimeString()}
                </p>
              )}
            </div>

            <div className="dashboard-header-actions">
              <button
                className={`dashboard-bell-button ${showNotifications ? 'open' : ''}`}
                onClick={handleToggleNotifications}
                aria-label="Notificações"
              >
                <FaBell size={18} />
                {unreadCount > 0 && <span className="dashboard-bell-badge">{unreadCount}</span>}
              </button>

              {showNotifications && (
                <div className="dashboard-notifications" ref={notificationsRef}>
                  <div className="dashboard-notifications-header">
                    <div>
                      <h3>Notificações</h3>
                      <span className="dashboard-notifications-subtitle">
                        {unreadCount > 0 ? `${unreadCount} não lida(s)` : 'Tudo em dia'}
                      </span>
                    </div>
                    {notificationItems.length > 0 && (
                      <button className="dashboard-notifications-mark" onClick={handleMarkAllRead}>
                        <FaCheck size={12} /> Marcar todas como lidas
                      </button>
                    )}
                  </div>

                  <div className="dashboard-notifications-list">
                    {notificationItems.length === 0 ? (
                      <div className="dashboard-notifications-empty">
                        Nenhuma notificação no momento.
                      </div>
                    ) : (
                      notificationItems.map((item) => {
                        const isRead = readNotifications.includes(item.id);
                        const formattedDate = item.date
                          ? item.date.toLocaleString()
                          : item.rawDate || 'Agora';

                        return (
                          <div
                            key={item.id}
                            className={`dashboard-notification-item ${isRead ? 'read' : 'unread'} notification-${item.type}`}
                          >
                            <div className="dashboard-notification-dot" />
                            <div className="dashboard-notification-content">
                              <strong>{item.title}</strong>
                              <p>{item.message}</p>
                              <span>{formattedDate}</span>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Cards Grid */}
      <div className="dashboard-stats">
        <div className="dashboard-grid">
          {/* Card 1: Atividades Recentes */}
          <div className="dashboard-card" style={{ gridColumn: '1 / -1' }}>
            <h3 className="dashboard-card-title">Atividades Recentes</h3>
            <div className="dashboard-activities">
              {filteredActivities.map((activity) => (
                <div key={activity.id} className="dashboard-activity">
                  <div className="dashboard-activity-icon">
                    {activity.icon === 'warning' ? (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
                      </svg>
                    ) : activity.icon === 'herd' ? (
                      <FaTractor size={18} color="#ffffff" />
                    ) : activity.icon === 'scale' ? (
                      <FaWeightHanging size={18} color="#ffffff" />
                    ) : (
                      <img 
                        src={`/images/${activity.icon}.png`} 
                        alt={activity.type} 
                        style={{width: '18px', height: '18px'}} 
                      />
                    )}
                  </div>
                  <div className="dashboard-activity-content">
                    <span className="dashboard-activity-text">{activity.description}</span>
                    <span className="dashboard-activity-date">
                      {(() => {
                        const dt = activity.date || activity.created_at || '';
                        if (!dt) return 'Sem data';
                        try {
                          const d = new Date(dt);
                          if (!isNaN(d.getTime())) return d.toLocaleString();
                        } catch (e) {
                          // fallthrough
                        }
                        return String(dt);
                      })()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Card 2: Cadastro de Gado */}
          <div className="dashboard-card" style={{ gridColumn: '1 / -1' }}>
            <h3 className="dashboard-card-title">Cadastro de Gado</h3>
            <div className="dashboard-registration-content">
              <div className="dashboard-registration-info">
                <div className="dashboard-registration-item">
                  <span className="label">Total cadastrados:</span>
                  <span className="value"> {dashboardData.totalCattle}</span>
                </div>
                <div className="dashboard-registration-item">
                  <span className="label">Último cadastro:</span>
                  <span className="value"> 
                    {(() => {
                      const v = dashboardData.lastRegistration;
                      if (!v) return '—';
                      try {
                        const d = new Date(v);
                        if (!isNaN(d.getTime())) return d.toLocaleString();
                      } catch (e) {}
                      return String(v);
                    })()}
                  </span>
                </div>
                <div className="dashboard-registration-item">
                  <span className="label">Fazenda ativa:</span>
                  <span className="value"> {dashboardData.activeFarm || currentUser?.farm_name || '—'}</span>
                </div>
              </div>
              <div className="dashboard-registration-icon">
                <img src="/images/vaca.png" alt="Vaca" style={{width: '48px', height: '48px'}} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Add Cattle Modal */}
      {showAddForm && (
        <div className="dashboard-modal">
          <div className="dashboard-modal-content">
            <h3 className="dashboard-modal-title">
              Adicionar Novo Bovino
            </h3>
            <form onSubmit={handleAddCattle} className="dashboard-form">
              <div className="dashboard-form-group">
                <label className="dashboard-form-label">Nome</label>
                <input
                  type="text"
                  value={newCattle.name}
                  onChange={(e) => setNewCattle({...newCattle, name: e.target.value})}
                  className="dashboard-form-input"
                  required
                />
              </div>
              <div className="dashboard-form-group">
                <label className="dashboard-form-label">Raça</label>
                <input
                  type="text"
                  value={newCattle.breed}
                  onChange={(e) => setNewCattle({...newCattle, breed: e.target.value})}
                  className="dashboard-form-input"
                  required
                />
              </div>
              <div className="dashboard-form-group">
                <label className="dashboard-form-label">Idade</label>
                <input
                  type="number"
                  value={newCattle.age}
                  onChange={(e) => setNewCattle({...newCattle, age: e.target.value})}
                  className="dashboard-form-input"
                  required
                />
              </div>
              <div className="dashboard-form-group">
                <label className="dashboard-form-label">Peso (kg)</label>
                <input
                  type="number"
                  value={newCattle.weight}
                  onChange={(e) => setNewCattle({...newCattle, weight: e.target.value})}
                  className="dashboard-form-input"
                  required
                />
              </div>
              <div className="dashboard-form-group">
                <label className="dashboard-form-label">Status de Saúde</label>
                <select
                  value={newCattle.healthStatus}
                  onChange={(e) => setNewCattle({...newCattle, healthStatus: e.target.value})}
                  className="dashboard-form-input"
                >
                  <option value="healthy">Saudável</option>
                  <option value="observation">Em Observação</option>
                  <option value="sick">Doente</option>
                </select>
              </div>
              <div className="dashboard-form-actions">
                <button
                  type="button"
                  onClick={() => setShowAddForm(false)}
                  className="dashboard-form-button cancel"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  className="dashboard-form-button submit"
                >
                  Adicionar
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
