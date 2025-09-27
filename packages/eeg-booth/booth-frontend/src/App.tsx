import React, { useState, useEffect, useCallback, useRef } from 'react';
import QRCode from 'qrcode';
import EEGVisualization from './components/EEGVisualization';
import AnalysisInterface from './components/AnalysisInterface';
import './App.css';

interface BoothStatus {
  booth_id: string;
  is_connected: boolean;
  scanner_connected: boolean;
  connection_status: string;
  timestamp: string;
}

interface BoothInfo {
  booth_id: string;
  qr_data: {
    booth_id: string;
    relayer_url: string;
  };
  status: string;
}

function App() {
  const [boothStatus, setBoothStatus] = useState<BoothStatus | null>(null);
  const [boothInfo, setBoothInfo] = useState<BoothInfo | null>(null);
  const [qrCodeDataUrl, setQrCodeDataUrl] = useState<string>('');
  const [loading, setLoading] = useState(true);
  
  // EEG WebSocket connection
  const [eegWebSocket, setEegWebSocket] = useState<WebSocket | null>(null);
  const [eegConnected, setEegConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const generateQRCode = async (data: string): Promise<string> => {
    try {
      const qrDataUrl = await QRCode.toDataURL(data, {
        width: 300,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#FFFFFF'
        }
      });
      return qrDataUrl;
    } catch (error) {
      console.error('Error generating QR code:', error);
      return '';
    }
  };

  const fetchBoothInfo = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:3004/booth-info');
      if (response.ok) {
        const data: BoothInfo = await response.json();
        setBoothInfo(data);
        
        // Generate QR code with booth connection data
        const qrData = JSON.stringify({
          booth_id: data.booth_id,
          relayer_url: 'wss://172.24.244.146:8765'
        });
        
        const qrDataUrl = await generateQRCode(qrData);
        setQrCodeDataUrl(qrDataUrl);
      }
    } catch (error) {
      console.error('Error fetching booth info:', error);
    }
  }, []);

    const fetchBoothStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:3004/status');
      if (response.ok) {
        const data: BoothStatus = await response.json();
        setBoothStatus(data);
      }
    } catch (error) {
      console.error('Error fetching booth status:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // EEG WebSocket connection management
  const connectToEEG = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    console.log('Connecting to EEG WebSocket server...');
    const ws = new WebSocket('ws://localhost:3005');
    
    ws.onopen = () => {
      console.log('✓ Connected to EEG server');
      setEegConnected(true);
      setEegWebSocket(ws);
      wsRef.current = ws;
    };
    
    ws.onclose = () => {
      console.log('EEG WebSocket disconnected');
      setEegConnected(false);
      setEegWebSocket(null);
      wsRef.current = null;
    };
    
    ws.onerror = (error) => {
      console.error('EEG WebSocket error:', error);
      setEegConnected(false);
    };

    return ws;
  }, []);

  const disconnectFromEEG = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setEegWebSocket(null);
      setEegConnected(false);
    }
  }, []);

  useEffect(() => {
    fetchBoothInfo();
    fetchBoothStatus();

    // Poll for status updates every 2 seconds
    const statusInterval = setInterval(fetchBoothStatus, 2000);

    return () => {
      clearInterval(statusInterval);
      disconnectFromEEG();
    };
  }, [fetchBoothInfo, fetchBoothStatus, disconnectFromEEG]);

  // Connect to EEG when user connects
  useEffect(() => {
    if (boothStatus?.connection_status === 'user_connected' && !eegConnected) {
      // Wait a moment for the booth connection to stabilize, then connect to EEG
      const timer = setTimeout(() => {
        connectToEEG();
      }, 2000);
      return () => clearTimeout(timer);
    } else if (boothStatus?.connection_status !== 'user_connected' && eegConnected) {
      // Disconnect from EEG when user disconnects
      disconnectFromEEG();
    }
  }, [boothStatus?.connection_status, eegConnected, connectToEEG, disconnectFromEEG]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'user_connected':
        return '#10B981'; // Green
      case 'registered':
      case 'connected':
        return '#3B82F6'; // Blue
      case 'disconnected':
      case 'connection_failed':
        return '#EF4444'; // Red
      case 'error':
        return '#F59E0B'; // Orange
      default:
        return '#6B7280'; // Gray
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'user_connected':
        return 'User Connected';
      case 'registered':
        return 'Ready for Connection';
      case 'connected':
        return 'Connecting...';
      case 'disconnected':
        return 'Disconnected';
      case 'connection_failed':
        return 'Connection Failed';
      case 'error':
        return 'Error';
      default:
        return 'Unknown Status';
    }
  };

  if (loading) {
    return (
      <div className="App">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Connecting to EEG Booth...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="booth-header">
        <div className="brain-icon">🧠</div>
        <h1>EEG Booth Station</h1>
        <div className="booth-id">
          Booth ID: {boothInfo?.booth_id || 'Unknown'}
        </div>
      </header>

      <main className="booth-main">
        <div className="status-section">
          <div 
            className="status-indicator"
            style={{ backgroundColor: getStatusColor(boothStatus?.connection_status || '') }}
          >
            <div className="status-dot"></div>
            <span>{getStatusText(boothStatus?.connection_status || '')}</span>
          </div>
        </div>

        {boothStatus?.connection_status === 'user_connected' ? (
          <div className="eeg-session">
            <div className="session-header">
              <div className="connected-icon">✅</div>
              <h2>EEG Session Active</h2>
              <div className="connection-status">
                <span className={`eeg-status ${eegConnected ? 'connected' : 'connecting'}`}>
                  EEG: {eegConnected ? 'Connected' : 'Connecting...'}
                </span>
                <span className="session-time">
                  Started: {new Date(boothStatus.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
            
            {eegConnected ? (
              <>
                <EEGVisualization websocket={eegWebSocket} isConnected={eegConnected} />
                <AnalysisInterface isConnected={eegConnected} />
              </>
            ) : (
              <div className="eeg-connecting">
                <div className="loading-spinner"></div>
                <p>Initializing EEG connection...</p>
                <p className="eeg-note">Connecting to OpenBCI hardware at ws://localhost:8765</p>
              </div>
            )}
          </div>
        ) : (
          <div className="qr-section">
            <div className="qr-container">
              <div className="brain-icon">🧠</div>
              <h2>EEG Booth Verification</h2>
              <p style={{margin: '0 0 32px 0', color: '#64748b', fontSize: '15px'}}>
                Connect your scanner to begin brain activity monitoring
              </p>
              
              <div className="privacy-notice">
                <div className="privacy-icon">🔒</div>
                <div className="privacy-content">
                  <h3>Privacy Protected</h3>
                  <p>No personal data stored on-chain</p>
                </div>
              </div>

              <div className="privacy-notice">
                <div className="privacy-icon">🧠</div>
                <div className="privacy-content">
                  <h3>Real-time Processing</h3>
                  <p>Scientific EEG analysis with immediate results</p>
                </div>
              </div>

              {qrCodeDataUrl ? (
                <div className="qr-display">
                  <img src={qrCodeDataUrl} alt="Booth QR Code" className="qr-code" />
                  <div className="qr-instructions">
                    <p>Scan with EEG Scanner</p>
                    <p className="qr-subtitle">Connect to begin your session</p>
                  </div>
                </div>
              ) : (
                <div className="qr-loading">
                  <div className="loading-spinner"></div>
                  <p>Generating connection code...</p>
                </div>
              )}
            </div>
          </div>
        )}

        <footer className="booth-footer">
          <div className="connection-details">
            <p>Status: {boothStatus?.is_connected ? 'Connected to Relayer' : 'Disconnected'}</p>
            <p>Last Update: {boothStatus ? new Date(boothStatus.timestamp).toLocaleTimeString() : 'Never'}</p>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
