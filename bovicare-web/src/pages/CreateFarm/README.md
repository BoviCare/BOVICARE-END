# Página Criar Fazenda

Esta página implementa a funcionalidade completa para criação de fazendas/rebanhos no sistema BoviCare.

## Funcionalidades Implementadas

### ✅ Campos do Formulário
- **Nome da fazenda** (obrigatório)
- **Cidade situada ou próxima** (obrigatório)
- **Localização da fazenda** (obrigatório) - com integração Google Maps
- **Área total da fazenda** (em hectares)
- **Capacidade de animais**
- **Nome do proprietário**
- **Número de funcionários**
- **Descrição da fazenda**

### ✅ Integração Google Maps
- Modal interativo para seleção de localização
- Clique no mapa para marcar a localização
- Reverse geocoding para obter endereço completo
- Fallback para coordenadas se endereço não for encontrado

### ✅ Upload de Documentos
- Upload múltiplo de arquivos
- Suporte a PDF, DOC, DOCX, JPG, JPEG, PNG
- Lista de arquivos anexados
- Opção de remover arquivos

### ✅ Validação e Feedback
- Validação de campos obrigatórios
- Mensagens de erro e sucesso
- Loading state durante criação
- Redirecionamento automático após sucesso

### ✅ Integração com Backend
- Função `createHerd()` conectada à API `/api/v1/herds`
- Tratamento de erros da API
- Mapeamento de dados do formulário para o modelo de dados

## Configuração Necessária

### Google Maps API Key
Para usar a funcionalidade de seleção de localização, configure a variável de ambiente:

```bash
# No arquivo .env
REACT_APP_GOOGLE_MAPS_API_KEY=sua_chave_api_google_maps
```

**Como obter a chave:**
1. Acesse [Google Cloud Console](https://console.cloud.google.com/)
2. Crie um novo projeto ou selecione um existente
3. Ative a API "Maps JavaScript API"
4. Crie credenciais (API Key)
5. Configure as restrições de domínio se necessário

## Estrutura dos Arquivos

```
src/pages/CreateFarm/
├── CreateFarm.js          # Componente principal
├── CreateFarm.css         # Estilos da página
└── README.md             # Esta documentação

src/components/GoogleMapSelector/
├── GoogleMapSelector.js   # Componente do Google Maps
└── GoogleMapSelector.css  # Estilos do modal do mapa
```

## Fluxo de Funcionamento

1. **Usuário acessa** `/criar-fazenda`
2. **Preenche o formulário** com os dados da fazenda
3. **Clica em "Selecionar"** para escolher localização no mapa
4. **Modal do Google Maps** abre para seleção interativa
5. **Confirma a localização** e volta ao formulário
6. **Anexa documentos** opcionais
7. **Submete o formulário** para criar a fazenda
8. **API processa** e salva no banco de dados
9. **Redirecionamento** para o dashboard após sucesso

## Dependências

- `@react-google-maps/api` - Integração com Google Maps
- `react-router-dom` - Navegação
- `axios` - Requisições HTTP (via services/api.js)

## Responsividade

A página é totalmente responsiva e se adapta a diferentes tamanhos de tela:
- Desktop: Layout em coluna única com largura máxima
- Tablet: Ajustes nos espaçamentos e tamanhos
- Mobile: Layout empilhado e botões em tela cheia
