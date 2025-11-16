import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="search-container">
        <input
          type="text"
          placeholder="Clique aqui para pesquisar"
          className="search-input"
        />
      </div>

      <div className="icons">
        <span>🛠️</span>
        <span>🔔</span>
        <span>⚙️</span>
      </div>
    </header>
  );
};

export default Header;
