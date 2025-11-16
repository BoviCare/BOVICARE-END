// Configurações de ambiente
const config = {
  development: {
    API_URL: 'http://localhost:5000',
    GOOGLE_MAPS_API_KEY: process.env.REACT_APP_GOOGLE_MAPS_API_KEY || '',
    TIMEOUT: 10000,
  },
  production: {
    API_URL: process.env.REACT_APP_API_URL || 'https://api.bovicare.com',
    GOOGLE_MAPS_API_KEY: process.env.REACT_APP_GOOGLE_MAPS_API_KEY || '',
    TIMEOUT: 15000,
  },
  test: {
    API_URL: 'http://localhost:5000',
    GOOGLE_MAPS_API_KEY: '',
    TIMEOUT: 5000,
  },
};

const environment = process.env.NODE_ENV || 'development';

export default config[environment];
