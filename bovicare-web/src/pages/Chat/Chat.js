import React, { useState, useRef, useEffect } from 'react';
import { FaUserMd, FaPaperPlane, FaRobot } from 'react-icons/fa';
import Navbar from '../../components/Navbar/Navbar';
import api from '../../services/api';
import './Chat.css';

const Chat = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Olá! Sou o assistente virtual do BoviCare. Como posso ajudá-lo hoje com questões sobre seu rebanho?",
      sender: 'assistant',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: messages.length + 1,
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const { data } = await api.post('/api/chat/diagnose', {
        message: userMessage.text
      });

      const assistantMessage = {
        id: userMessage.id + 1,
        text: data?.reply || 'Não consegui gerar uma resposta no momento.',
        sender: 'assistant',
        timestamp: new Date(),
        sources: data?.sources || []
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const assistantMessage = {
        id: userMessage.id + 1,
        text: 'Não consegui obter uma resposta. Tente novamente em instantes.',
        sender: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString('pt-BR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className="chat-container">
      <Navbar />
      <div className="chat-main">
        <div className="chat-header">
          <div className="chat-header-content">
            <div className="assistant-avatar">
              <img src="/images/assistenteBovicare.jpg" alt="Assistente BoviCare" />
            </div>
            <div className="chat-header-info">
              <h1>Chat</h1>
              <p>Assistente Virtual BoviCare</p>
            </div>
          </div>
        </div>

        <div className="chat-messages">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-avatar">
                {message.sender === 'user' ? <FaUserMd /> : <img src="/images/assistenteBovicare.jpg" alt="Assistente" />}
              </div>
              <div className="message-content">
                <div className="message-bubble">
                  <p>{message.text}</p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <strong>Fontes:</strong>
                      <ul>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>
                            {source.disease_name || 'Fonte'}
                            {source.section_type ? ` - ${source.section_type}` : ''}
                            {source.page_number ? ` (p. ${source.page_number})` : ''}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <span className="message-time">{formatTime(message.timestamp)}</span>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message assistant-message">
              <div className="message-avatar">
                <img src="/images/assistenteBovicare.jpg" alt="Assistente" />
              </div>
              <div className="message-content">
                <div className="message-bubble loading">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <form onSubmit={handleSendMessage}>
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Envie uma mensagem"
              disabled={isLoading}
              className="message-input"
            />
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chat;
