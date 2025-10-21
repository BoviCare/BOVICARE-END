import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { validateNewPassword } from '../../../services/validation';
import { resetPassword } from '../../../services/api';
import './forgot.css';

function ForgotNewPassword() {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [resetToken, setResetToken] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Verificar se existe token válido
    const token = localStorage.getItem('resetToken');
    if (!token) {
      navigate('/esqueci-senha');
      return;
    }
    setResetToken(token);
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const { isValid, errors: errs } = validateNewPassword({ password, confirmPassword });
    if (!isValid) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setLoading(true);

    try {
      await resetPassword({
        token: resetToken,
        new_password: password
      });
      
      // Limpar dados temporários
      localStorage.removeItem('resetData');
      localStorage.removeItem('resetToken');
      
      // Navegar para login com mensagem de sucesso
      navigate('/login?success=password-reset');
    } catch (error) {
      setErrors({ general: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="forgot-wrapper">
        <img src="/images/logo.svg" alt="Logo" />
        <div className="card forgot-card">
          <h2>Nova senha</h2>
          <span>Digite sua nova senha</span>
          
          <form onSubmit={handleSubmit}>
            <input
              type="password"
              placeholder="Nova senha"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            
            <input
              type="password"
              placeholder="Confirmar nova senha"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
            
            {(errors.password || errors.confirmPassword || errors.general) && (
              <p className="error-text">
                {errors.password || errors.confirmPassword || errors.general}
              </p>
            )}
            
            <button type="submit" disabled={loading}>
              {loading ? 'Redefinindo...' : 'Redefinir senha'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ForgotNewPassword;
