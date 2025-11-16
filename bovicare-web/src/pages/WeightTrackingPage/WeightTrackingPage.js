// Importando as dependências
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import Navbar from '../../components/Navbar/Navbar';
import WeightTracking from '../../components/WeightTracking/WeightTracking';
import './WeightTrackingPage.css';

const WeightTrackingPage = () => {
  const { user } = useAuth();
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
    // Implementar lógica de busca específica para acompanhamento de peso
  };

  return (
    <div className="weight-tracking-page-container">
      <Navbar />
      <WeightTracking />
    </div>
  );
};

export default WeightTrackingPage;
