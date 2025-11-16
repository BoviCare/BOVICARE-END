<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
</head>
<body>
    <h1>Bovicare</h1>
    <img src="logo.svg" width=100px>
    <p>Bovicare é um sistema web desenvolvido com Flask, ReactJS, Python, Docker e PostgreSQL. O objetivo do Bovicare é fornecer uma plataforma robusta e escalável para o gerenciamento de informações relacionadas ao cuidado de bovinos.</p>

  <h2>Tecnologias Utilizadas</h2>
    <ul>
        <li><img src="https://img.icons8.com/color/48/000000/react-native.png" alt="ReactJS logo"><strong>Frontend:</strong> ReactJS</li>
        <li><img src="https://img.icons8.com/ios-filled/50/000000/flask.png" alt="Flask logo"><strong>Backend:</strong> Flask (Python 3.11)</li>
        <li><img src="https://img.icons8.com/color/48/000000/postgreesql.png" alt="PostgreSQL logo"><strong>Banco de Dados:</strong> PostgreSQL</li>
        <li><img src="https://img.icons8.com/color/48/000000/docker.png" alt="Docker logo"><strong>Containerização:</strong> Docker</li>
        <li><img src="https://img.icons8.com/color/48/000000/python.png" alt="Python logo"><strong>Linguagem:</strong> Python</li>
    </ul>

  <h2>Funcionalidades do sistema</h2>
    <ul>
        <li>Gerenciamento de informações de bovinos</li>
        <li>Registro de dados de saúde e alimentação</li>
        <li>Acompanhamento de crescimento e produtividade</li>
        <li>Interface amigável e responsiva</li>
    </ul>

  <h2>Pré-requisitos</h2>
    <p>Antes de começar, você vai precisar ter instalado em sua máquina as seguintes ferramentas:</p>
    <ul>
        <li><a href="https://git-scm.com">Git</a></li>
        <li><a href="https://www.docker.com">Docker</a></li>
        <li><a href="https://docs.docker.com/compose/">Docker Compose</a></li>
        <li><a href="https://www.python.org">Python</a></li>
        <li><a href="https://nodejs.org">Node.js</a></li>
    </ul>

  <h2>Como rodar o projeto</h2>
    <p>Siga os passos abaixo para configurar e executar o projeto em seu ambiente de desenvolvimento:</p>
    <ol>
        <li>
            <p><strong>Clone o repositório</strong></p>
            <pre><code>git clone https://github.com/seu-usuario/bovicare.git</code></pre>
        </li>
        <li>
            <p><strong>Navegue até o diretório do projeto</strong></p>
            <pre><code>cd bovicare</code></pre>
        </li>
        <li>
            <p><strong>Configure as variáveis de ambiente</strong></p>
            <p>Crie um arquivo <code>.env</code> na raiz do projeto e adicione as seguintes variáveis:</p>
            <pre><code>
FLASK_APP=app
FLASK_ENV=development
DATABASE_URL=postgresql://usuario:senha@localhost:5432/bovicare
            </code></pre>
        </li>
        <li>
            <p><strong>Construa e inicie os containers Docker</strong></p>
            <pre><code>docker compose up --build</code></pre>
        </li>
        <li>
            <p><strong>Instale as dependências do frontend</strong></p>
            <pre><code>
cd frontend
npm install
npm start
            </code></pre>
        </li>
        <li>
            <p><strong>Acesse a aplicação</strong></p>
            <p>O backend estará disponível em <code>http://localhost:5000</code> e o frontend em <code>http://localhost:3000</code>.</p>
        </li>
    </ol>

  <h2>Estrutura do Projeto</h2>
    <pre><code>
bovicare/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── ...
│   ├── Dockerfile
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.js
│   │   ├── index.js
│   │   └── ...
│   ├── Dockerfile
│   ├── package.json
│   └── ...
│
├── docker-compose.yml
├── .env
└── README.md
    </code></pre>

  <h2>Licença</h2>
    <p>Este projeto está licenciado sob a licença MIT. Veja o arquivo <a href="LICENSE">LICENSE</a> para mais detalhes.</p>
</body>
</html>
