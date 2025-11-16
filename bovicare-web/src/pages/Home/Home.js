// Importando as dependências
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import Navbar from '../../components/Navbar/Navbar';
import Dashboard from '../../components/Dashboard/Dashboard';
import './Home.css';

const Home = () => {
  const { user } = useAuth();
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
    // Implementar lógica de busca
  };


  return (
    <div className="home-container">
      <Navbar />
      <Dashboard />
    </div>
  );
};

export default Home;
