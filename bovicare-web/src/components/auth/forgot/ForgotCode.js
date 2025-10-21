import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { isValidSixDigitCode } from '../../../services/validation';
import { verifyResetCode } from '../../../services/api';
import './forgot.css';

function ForgotCode() {
  const [values, setValues] = useState(['', '', '', '', '', '']);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [resetData, setResetData] = useState(null);
  const inputsRef = useRef([]);
  const navigate = useNavigate();

  useEffect(() => {
    // Recuperar dados da etapa anterior
    const data = localStorage.getItem('resetData');
    if (!data) {
      navigate('/esqueci-senha');
      return;
    }
    setResetData(JSON.parse(data));
  }, [navigate]);

  const handleChange = (index, val) => {
    if (!/^\d?$/.test(val)) return;
    const updated = [...values];
    updated[index] = val;
    setValues(updated);
    if (val && index < 5) {
      const next = inputsRef.current[index + 1];
      next && next.focus();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const code = values.join('');
    if (!isValidSixDigitCode(code)) {
      setError('Código deve conter 6 dígitos');
      return;
    }
    setError('');
    setLoading(true);

    try {
      const data = {
        code,
        method: resetData.method,
        ...(resetData.method === 'email' ? { email: resetData.email } : { phone: resetData.phone })
      };
      
      const response = await verifyResetCode(data);
      
      // Salvar token para próxima etapa
      localStorage.setItem('resetToken', response.token);
      
      // Navegar para tela de nova senha (vamos criar depois)
      navigate('/esqueci-senha/nova-senha');
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="forgot-wrapper">
        <img src="/images/logo.svg" alt="Logo" />
        <div className="card forgot-card">
          <h2>Redefina sua senha</h2>
          <span>Insira o código que te enviamos por e-mail</span>
          <form onSubmit={handleSubmit}>
            <div className="code-row">
              {values.map((v, i) => (
                <input
                  key={i}
                  ref={(el) => (inputsRef.current[i] = el)}
                  inputMode="numeric"
                  maxLength={1}
                  value={v}
                  onChange={(e) => handleChange(i, e.target.value)}
                  className="code-input"
                />
              ))}
            </div>
            {error && <p className="error-text">{error}</p>}
            <button type="submit" disabled={loading}>
              {loading ? 'Verificando...' : 'Confirmar código'}
            </button>
            <p className="muted">Reenviar código em 00:30</p>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ForgotCode;


