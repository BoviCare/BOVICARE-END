import React, { useState, useCallback } from 'react';
import { GoogleMap, useJsApiLoader, Marker } from '@react-google-maps/api';
import config from '../../config/environment';
import './GoogleMapSelector.css';

const mapContainerStyle = {
  width: '100%',
  height: '400px'
};

const defaultCenter = {
  lat: -15.7801, // Centro do Brasil (Goiás)
  lng: -47.9292
};

const GoogleMapSelector = ({ onLocationSelect, isOpen, onClose }) => {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [map, setMap] = useState(null);

  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: config.GOOGLE_MAPS_API_KEY,
    libraries: ['places']
  });

  const onLoad = useCallback((map) => {
    setMap(map);
  }, []);

  const onUnmount = useCallback(() => {
    setMap(null);
  }, []);

  const handleMapClick = (event) => {
    const lat = event.latLng.lat();
    const lng = event.latLng.lng();
    
    setSelectedLocation({
      lat: lat,
      lng: lng
    });
  };

  const handleConfirmLocation = () => {
    if (selectedLocation) {
      // Fazer reverse geocoding para obter o endereço
      const geocoder = new window.google.maps.Geocoder();
      
      geocoder.geocode(
        { location: selectedLocation },
        (results, status) => {
          if (status === 'OK' && results[0]) {
            const address = results[0].formatted_address;
            onLocationSelect(address, selectedLocation);
            onClose();
          } else {
            // Mostrar erro se não conseguir obter o endereço
            alert('Erro ao obter endereço. Tente selecionar uma localização mais específica.');
          }
        }
      );
    }
  };

  if (loadError) {
    return (
      <div className="map-modal-overlay">
        <div className="map-modal">
          <div className="map-error">
            <h3>⚠️ Configuração Necessária</h3>
            <p>Para usar o seletor de localização, é necessário configurar a chave da API do Google Maps.</p>
            <div className="error-steps">
              <h4>Como configurar:</h4>
              <ol>
                <li>Acesse <a href="https://console.cloud.google.com/" target="_blank" rel="noopener noreferrer">Google Cloud Console</a></li>
                <li>Crie um novo projeto ou selecione um existente</li>
                <li>Ative a API "Maps JavaScript API"</li>
                <li>Crie uma chave de API</li>
                <li>Configure as restrições de domínio (localhost:3000 para desenvolvimento)</li>
                <li>Crie um arquivo <code>.env</code> na raiz do projeto com:</li>
              </ol>
              <div className="code-block">
                <code>REACT_APP_GOOGLE_MAPS_API_KEY=sua_chave_aqui</code>
              </div>
            </div>
            <button className="close-btn" onClick={onClose}>Fechar</button>
          </div>
        </div>
      </div>
    );
  }

  if (!isLoaded) {
    return (
      <div className="map-modal-overlay">
        <div className="map-modal">
          <div className="map-loading">
            <div className="loading-spinner"></div>
            <p>Carregando mapa...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!isOpen) return null;

  return (
    <div className="map-modal-overlay" onClick={onClose}>
      <div className="map-modal" onClick={(e) => e.stopPropagation()}>
        <div className="map-modal-header">
          <h3>Selecionar Localização da Fazenda</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="map-container">
          <GoogleMap
            mapContainerStyle={mapContainerStyle}
            center={defaultCenter}
            zoom={10}
            onLoad={onLoad}
            onUnmount={onUnmount}
            onClick={handleMapClick}
          >
            {selectedLocation && (
              <Marker
                position={selectedLocation}
                title="Localização selecionada"
              />
            )}
          </GoogleMap>
        </div>
        
        <div className="map-instructions">
          <p>Clique no mapa para selecionar a localização da fazenda</p>
          {selectedLocation && (
            <div className="selected-coords">
              <p>Coordenadas selecionadas:</p>
              <p>Lat: {selectedLocation.lat.toFixed(6)}</p>
              <p>Lng: {selectedLocation.lng.toFixed(6)}</p>
            </div>
          )}
        </div>
        
        <div className="map-modal-actions">
          <button className="cancel-btn" onClick={onClose}>
            Cancelar
          </button>
          <button 
            className="confirm-btn" 
            onClick={handleConfirmLocation}
            disabled={!selectedLocation}
          >
            Confirmar Localização
          </button>
        </div>
      </div>
    </div>
  );
};

export default GoogleMapSelector;
