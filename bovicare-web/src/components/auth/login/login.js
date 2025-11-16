import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../../../services/api';
import './login.css';
import { useAuth } from '../../../contexts/AuthContext';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const emailRef = useRef(null);
  const navigate = useNavigate();
  const { loginUser } = useAuth();

  useEffect(() => {
    emailRef.current && emailRef.current.focus();
  }, []);

  const handleSubmit = async (e) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    // Limpar erro anterior
    setError('');

    // Verifica se os campos estão preenchidos
    if (!email || !password) {
      setError('Preencha todos os campos');
      return false;
    }

    try {
      console.log('Tentando login com:', email);
      // Usar o AuthContext para garantir atualização do estado global
      const response = await loginUser({ email, password });
      console.log('Resposta do login:', response);

      if (response && response.success) {
        console.log('Login bem-sucedido, navegando para /home');
        navigate('/home');
      } else {
        console.log('Login falhou - credenciais incorretas');
        setError('Email ou senha incorretos');
        emailRef.current && emailRef.current.focus();
      }

      return false;
    } catch (err) {
      console.error('Erro no login:', err);
      setError('Email ou senha incorretos');
      emailRef.current && emailRef.current.focus();
      return false;
    }
  };

  // Garantir caminho correto para imagens em todos os navegadores
  const publicUrl = process.env.PUBLIC_URL || '';
  const logoPath = publicUrl ? `${publicUrl}/images/logo.svg` : '/images/logo.svg';
  const backgroundPath = publicUrl ? `${publicUrl}/images/fundo.svg` : '/images/fundo.svg';
  
  return (
    <div className="login-container" style={{ backgroundImage: `url(${backgroundPath})` }}>
      <img src={logoPath} alt="Logo" />
      <div className="card">
        <h2>Entre na sua conta</h2>
        <span>Preencha os campos abaixo</span>
        <form onSubmit={handleSubmit}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            ref={emailRef}
          />
          <input
            type="password"
            placeholder="Senha"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit">Entrar</button>
        </form>
        {error && (
          <div style={{
            background: '#ffebee',
            color: '#c62828',
            padding: '14px 18px',
            borderRadius: '8px',
            marginTop: '16px',
            marginBottom: '8px',
            fontSize: '0.95rem',
            fontWeight: '600',
            border: '1px solid #ef5350',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}
        <p>
          <a href="/esqueci-senha">Esqueceu sua senha?</a>
        </p>
      </div>
    </div>
  );
}

export default Login;
