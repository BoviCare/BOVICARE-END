import React, { useState, useRef, useEffect } from 'react';
import { registerUser } from '../../../services/api';
import './register.css';

function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Referências para os campos de nome, email e senha
  const nameRef = useRef(null);
  const emailRef = useRef(null);
  const passwordRef = useRef(null);

  // Focar no campo de nome quando o componente for montado
  useEffect(() => {
    nameRef.current && nameRef.current.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Chama a função do serviço para registrar o usuário
      await registerUser({ username: name, email, password });
      setSuccess('Usuário cadastrado com sucesso!');
      setEmail('');
      setPassword('');
      setName('');
      setError('');
      nameRef.current && nameRef.current.focus(); // Focar no campo de nome após o sucesso
    } catch (err) {
      setError('Erro ao cadastrar usuário');
      setSuccess('');
      nameRef.current && nameRef.current.focus(); // Focar no campo de nome após o erro
    }
  };

  return (
    <div className="register-container">
      <img src="/images/logo.svg" alt="Logo" />
    <div className="card">
      <h2>Cadastrar</h2>
      <span>Preencha os campos abaixo</span>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Nome"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          ref={nameRef} // Atribuindo a referência ao campo de nome
        />
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          ref={emailRef} // Atribuindo a referência ao campo de email
        />
        <input
          type="password"
          placeholder="Senha"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          ref={passwordRef} // Atribuindo a referência ao campo de senha
        />
        <button type="submit">Cadastrar</button>
      </form>

      {error && <p>{error}</p>}
      {success && <p>{success}</p>}

      {/* Texto para redirecionar para a tela de login */}
      <p>
        Já tem uma conta? <a href="/login">Faça login</a>
      </p>
    </div>

      {error && <p>{error}</p>}
      {success && <p>{success}</p>}
    </div>
  );
}

export default Register;
