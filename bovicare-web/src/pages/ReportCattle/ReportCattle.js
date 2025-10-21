import React, { useEffect, useState } from 'react';
import Navbar from '../../components/Navbar/Navbar';
import { getWeightReport } from '../../services/api';
import './ReportCattle.css';

const statusColorMap = {
  'Meta atingida': 'status-success',
  'Alerta: perda de peso': 'status-danger',
  'Em progresso': 'status-warning',
  'Sem dados suficientes': 'status-muted',
  'Acompanhamento em progresso': 'status-info'
};

const trendIconMap = {
  up: '▲',
  down: '▼',
  stable: '■'
};

const ReportCattle = () => {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getWeightReport();
      setReport(data);
    } catch (err) {
      setError(err.message || 'Erro ao gerar relatório.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReport();
  }, []);

  const renderSummaryCard = (title, value, description) => (
    <div className="report-summary-card">
      <div className="report-summary-title">{title}</div>
      <div className="report-summary-value">{value ?? '—'}</div>
      {description && <div className="report-summary-description">{description}</div>}
    </div>
  );

  const renderStatusPill = (status) => {
    const normalized = statusColorMap[status] ? status : 'Sem dados suficientes';
    return <span className={`status-pill ${statusColorMap[normalized] || statusColorMap['Sem dados suficientes']}`}>{status}</span>;
  };

  if (loading) {
    return (
      <div className="report-cattle-container">
        <Navbar />
        <div className="report-content loading-state">
          <div className="spinner" />
          <p>Gerando relatório automático...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="report-cattle-container">
        <Navbar />
        <div className="report-content error-state">
          <p>{error}</p>
          <button onClick={loadReport} className="reload-button">Tentar novamente</button>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="report-cattle-container">
        <Navbar />
        <div className="report-content empty-state">
          <p>Nenhum dado disponível para o relatório.</p>
        </div>
      </div>
    );
  }

  const { summary, alerts, animals } = report;

  return (
    <div className="report-cattle-container">
      <Navbar />
      <div className="report-content">
        <header className="report-header">
          <div>
            <h1>Relatório de Peso dos Gados</h1>
            <p>Acompanhamento automático baseado nas metas de peso definidas para cada animal.</p>
          </div>
          <button className="reload-button" onClick={loadReport}>Atualizar relatório</button>
        </header>

        <section className="report-summary">
          {renderSummaryCard('Total de gados monitorados', summary.totalCattle, 'Animais com pelo menos uma pesagem registrada.')}
          {renderSummaryCard('Com meta cadastrada', summary.withTarget, 'Animais que possuem objetivo de peso definido.')}
          {renderSummaryCard('Meta atingida', summary.reachedTarget, 'Animais que já alcançaram o peso de abate.')}
          {renderSummaryCard('Perdendo peso', summary.losingWeight, 'Animais com tendência de queda e que exigem atenção imediata.')}
          {renderSummaryCard('Sem dados de peso', summary.withoutData, 'Animais sem pesagens registradas ou sem meta definida.')}
          {renderSummaryCard('Peso médio atual', summary.averageWeight ? `${summary.averageWeight} kg` : '—', 'Média do peso atual considerando animais com pesagem recente.')}
        </section>

        <section className="report-alerts">
          <div className="section-header">
            <h2>Alertas prioritários</h2>
            <span>{alerts.length} alerta(s)</span>
          </div>
          {alerts.length === 0 ? (
            <div className="alert-empty">Nenhum alerta no momento. Continue monitorando o rebanho.</div>
          ) : (
            <div className="alert-grid">
              {alerts.map((animal) => (
                <div key={animal.id} className="alert-card">
                  <div className="alert-card-header">
                    <span className="alert-name">{animal.name}</span>
                    {renderStatusPill(animal.status)}
                  </div>
                  <div className="alert-details">
                    <div>
                      <span className="info-label">Peso atual</span>
                      <span className="info-value">{animal.currentWeight ?? '—'} kg</span>
                    </div>
                    <div>
                      <span className="info-label">Meta</span>
                      <span className="info-value">{animal.targetWeight ?? '—'} kg</span>
                    </div>
                    <div>
                      <span className="info-label">Diferença para meta</span>
                      <span className="info-value">{animal.differenceToTarget != null ? `${animal.differenceToTarget} kg` : '—'}</span>
                    </div>
                    <div>
                      <span className="info-label">Última pesagem</span>
                      <span className="info-value">{animal.lastWeighingDate ? new Date(animal.lastWeighingDate).toLocaleDateString('pt-BR') : '—'}</span>
                    </div>
                  </div>
                  <p className="alert-message">{animal.message}</p>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="report-animal-list">
          <div className="section-header">
            <h2>Detalhes por animal</h2>
            <span>{animals.length} registro(s)</span>
          </div>
          <div className="animal-table">
            <div className="animal-table-header">
              <span>Animal</span>
              <span>Peso atual</span>
              <span>Meta</span>
              <span>Diferença</span>
              <span>% atingido</span>
              <span>Tendência</span>
              <span>Status</span>
            </div>
            {animals.map((animal) => (
              <div key={animal.id} className="animal-table-row">
                <span className="animal-name">{animal.name}</span>
                <span>{animal.currentWeight != null ? `${animal.currentWeight} kg` : '—'}</span>
                <span>{animal.targetWeight != null ? `${animal.targetWeight} kg` : '—'}</span>
                <span>{animal.differenceToTarget != null ? `${animal.differenceToTarget} kg` : '—'}</span>
                <span>{animal.percentageToTarget != null ? `${animal.percentageToTarget}%` : '—'}</span>
                <span className={`trend trend-${animal.trend}`}>{trendIconMap[animal.trend] || '■'}</span>
                <span>{renderStatusPill(animal.status)}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default ReportCattle;
