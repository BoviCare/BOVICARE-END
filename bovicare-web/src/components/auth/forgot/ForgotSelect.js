import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { validateForgotPasswordSelection } from '../../../services/validation';
import { forgotPassword } from '../../../services/api';
import './forgot.css';

function ForgotSelect() {
  const [email, setEmail] = useState('');
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleConfirm = async (e) => {
    e.preventDefault();
    const { isValid, errors: errs } = validateForgotPasswordSelection({ method: 'email', email });
    if (!isValid) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setLoading(true);

    try {
      await forgotPassword({ method: 'email', email });
      localStorage.setItem('resetData', JSON.stringify({ method: 'email', email }));
      navigate('/esqueci-senha/codigo');
    } catch (error) {
      setErrors({ general: error.message });
    } finally {
      setLoading(false);
    }
  };

  // Garantir caminho correto para imagens em todos os navegadores
  const publicUrl = process.env.PUBLIC_URL || '';
  const logoPath = publicUrl ? `${publicUrl}/images/logo.svg` : '/images/logo.svg';
  const backgroundPath = publicUrl ? `${publicUrl}/images/fundo.svg` : '/images/fundo.svg';
  
  return (
    <div className="login-container" style={{ backgroundImage: `url(${backgroundPath})` }}>
      <div className="forgot-wrapper">
        <img src={logoPath} alt="Logo" />
        <div className="card forgot-card">
          <h2>Redefina sua senha</h2>
          <span>Informe seu e-mail para receber um código de verificação</span>

          <form onSubmit={handleConfirm}>
            <input
              type="email"
              placeholder="Digite seu e-mail"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />

            {(errors.email || errors.general) && (
              <p className="error-text">{errors.email || errors.general}</p>
            )}

            <button type="submit" disabled={loading}>
              {loading ? 'Enviando...' : 'Confirmar'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ForgotSelect;


