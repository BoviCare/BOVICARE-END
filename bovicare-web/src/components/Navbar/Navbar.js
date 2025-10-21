import React, { useState } from 'react';
import { FaHome, FaUser, FaBell, FaPlus, FaMapMarkedAlt, FaSignOutAlt, FaCog, FaWeight, FaFileAlt, FaComments, FaTimes, FaArrowLeft, FaBars, FaUsers } from 'react-icons/fa';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import './Navbar.css';

const Navbar = () => {
  const { logoutUser, user } = useAuth();
  const [isClosed, setIsClosed] = useState(false);
  
  // Verificar se o usuário é admin
  const isAdmin = user?.role === 'admin';

  const handleLogout = () => {
    logoutUser();
  };

  const handleCloseSidebar = () => {
    setIsClosed(true);
  };

  const handleOpenSidebar = () => {
    setIsClosed(false);
  };

  return (
    <>
      {/* Botão para abrir sidebar quando fechado */}
      <button 
        className={`sidebar-toggle ${isClosed ? 'visible' : ''}`}
        onClick={handleOpenSidebar}
        title="Abrir menu"
      >
        <FaBars />
      </button>

      <div className={`navbar ${isClosed ? 'closed' : ''}`}>
      <div className="navbar-logo">
        <img src="/images/logo.svg" alt="BoviCare Logo" className="logo-img" />
        <span className="logo-text">BoviCare</span>
      </div>

      <div className="navbar-section">
        <div className="navbar-section-title">Navegação</div>
        <ul className="navbar-menu">
          <li>
            <NavLink to="/home" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaHome /> Início
            </NavLink>
          </li>
          <li>
            <NavLink to="/profile" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaUser /> Perfil
            </NavLink>
          </li>
        </ul>
      </div>

      <div className="navbar-section">
        <div className="navbar-section-title">Funcionalidades</div>
        <ul className="navbar-menu">
          <li>
            <NavLink to="/acompanhar-peso" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaWeight /> Acompanhar peso
            </NavLink>
          </li>
          <li>
            <NavLink to="/relatorio-gados" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaFileAlt /> Relatório gados
            </NavLink>
          </li>
          <li>
            <NavLink to="/criar-fazenda" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaPlus /> Criar fazenda
            </NavLink>
          </li>
          <li>
            <NavLink to="/cadastrar-gado" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaPlus /> Cadastrar gado
            </NavLink>
          </li>
          <li>
            <NavLink to="/chat" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <FaComments /> Chat...
            </NavLink>
          </li>
        </ul>
      </div>

      {/* Seção de Administração - apenas para admins */}
      {isAdmin && (
        <div className="navbar-section">
          <div className="navbar-section-title">Administração</div>
          <ul className="navbar-menu">
            <li>
              <NavLink to="/usuarios" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
                <FaUsers /> Usuários
              </NavLink>
            </li>
          </ul>
        </div>
      )}

      <div className="bottom-link">
        <ul className="navbar-menu">
          <li>
            <button 
              className="nav-item"
              onClick={handleCloseSidebar}
              style={{ background: 'none', border: 'none', width: '100%', textAlign: 'left', cursor: 'pointer' }}
            >
              <FaTimes /> Fechar aba lateral
            </button>
          </li>
          <li>
            <button 
              onClick={handleLogout}
              className="nav-item"
              style={{ background: 'none', border: 'none', width: '100%', textAlign: 'left', cursor: 'pointer' }}
            >
              <FaArrowLeft /> Sair
            </button>
          </li>
        </ul>
      </div>
      </div>
    </>
  );
};

export default Navbar;
