import asyncio
import websockets
import json
import logging
import uuid
import threading
import ssl
import os
import time
import queue
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
try:
    import numpy as np
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
    from eeg_processor import EEGProcessor
    BRAINFLOW_AVAILABLE = True
    EEG_AVAILABLE = True
except ImportError:
    print("Warning: BrainFlow not available. Install brainflow for OpenBCI support.")
    BRAINFLOW_AVAILABLE = False
    EEG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoothBackend:
    def __init__(self, booth_id=None, relayer_url="wss://172.24.244.146:8765", frontend_port=3004):
        self.booth_id = booth_id or f"booth_{uuid.uuid4().hex[:8]}"
        self.relayer_url = relayer_url
        self.frontend_port = frontend_port
        self.websocket = None
        self.is_connected = False
        self.scanner_connected = False
        self.connection_status = "disconnected"
        
        # EEG WebSocket server for frontend
        self.eeg_clients = set()
        self.eeg_server_port = 3005
        
        # BrainFlow OpenBCI connection
        self.board_id = BoardIds.CYTON_BOARD.value  # OpenBCI Cyton 8-channel
        self.board_shim = None
        self.sampling_rate = 250  # OpenBCI Cyton sampling rate
        self.eeg_streaming = False
        self.eeg_data_queue = queue.Queue(maxsize=1000)
        
        # BrainFlow parameters
        self.params = BrainFlowInputParams()
        self.params.serial_port = "/dev/cu.usbserial-DM01MV82"  # Update with your port
        
        # EEG processor
        if EEG_AVAILABLE:
            self.eeg_processor = EEGProcessor(sampling_rate=250)
        else:
            self.eeg_processor = None
            
        # Store recent EEG data for frequency analysis (2 seconds = 500 samples)
        self.recent_channel_data = [[] for _ in range(8)]
        self.max_recent_samples = 500
        
        # Flask app for serving frontend data
        self.app = Flask(__name__)
        CORS(self.app)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for frontend communication"""
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            return jsonify({
                'booth_id': self.booth_id,
                'is_connected': self.is_connected,
                'scanner_connected': self.scanner_connected,
                'connection_status': self.connection_status,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/booth-info', methods=['GET'])
        def get_booth_info():
            return jsonify({
                'booth_id': self.booth_id,
                'qr_data': {
                    'booth_id': self.booth_id,
                    'relayer_url': self.relayer_url.replace('ws://', 'wss://').replace('localhost', '127.0.0.1')
                },
                'status': self.connection_status
            })
        
        @self.app.route('/send-message', methods=['POST'])
        def send_message():
            # Future endpoint for sending messages to scanner
            return jsonify({'status': 'message_sent'})
        
        @self.app.route('/analyze-personality', methods=['POST'])
        def analyze_personality():
            """Analyze Big 5 personality traits from current EEG data"""
            if not self.eeg_processor or not EEG_AVAILABLE:
                return jsonify({'error': 'EEG processor not available'}), 400
                
            if not self.eeg_streaming or len(self.recent_channel_data[0]) < 500:
                return jsonify({'error': 'Insufficient EEG data for analysis -> ' + str(len(self.recent_channel_data[0])) + ' samples available'}), 400

            try:
                logger.info("🧠 Starting Big 5 personality analysis...")
                
                # Use recent EEG data for personality analysis
                personality_result = self.eeg_processor.calculate_big5_personality(self.recent_channel_data)
                
                if personality_result:
                    logger.info(f"✅ Personality analysis complete")
                    return jsonify({
                        'status': 'success',
                        'analysis_type': 'big5_personality',
                        'personality': personality_result,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Personality analysis failed'}), 400
                    
            except Exception as e:
                logger.error(f"❌ Personality analysis error: {e}")
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
        @self.app.route('/analyze-dating-preference', methods=['POST'])
        def analyze_dating_preference():
            """Analyze dating preference based on EEG response to stimulus"""
            if not self.eeg_processor or not EEG_AVAILABLE:
                return jsonify({'error': 'EEG processor not available'}), 400
                
            if not self.eeg_streaming or len(self.recent_channel_data[0]) < 250:
                return jsonify({'error': 'Insufficient EEG data for analysis'}), 400
            
            try:
                data = request.get_json()
                stimulus_type = data.get('stimulus_type', 'unknown')
                stimulus_duration = data.get('duration', 3)
                
                logger.info(f"💕 Analyzing dating preference for stimulus: {stimulus_type}")
                
                # Analyze EEG response to the stimulus
                preference_result = self.eeg_processor.analyze_dating_preference(
                    self.recent_channel_data, 
                    stimulus_type, 
                    stimulus_duration
                )
                
                if preference_result:
                    logger.info(f"✅ Dating preference analysis complete: {preference_result['attraction_level']}")
                    return jsonify({
                        'status': 'success',
                        'analysis_type': 'dating_preference',
                        'preference': preference_result,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'Dating preference analysis failed'}), 400
                    
            except Exception as e:
                logger.error(f"❌ Dating preference analysis error: {e}")
                return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
        @self.app.route('/send-analysis-results', methods=['POST'])
        def send_analysis_results():
            """Send complete EEG analysis results to scanner frontend"""
            try:
                data = request.get_json()
                
                # Create comprehensive results message
                results_message = {
                    'type': 'eeg_analysis_complete',
                    'booth_id': self.booth_id,
                    'personality': data.get('personality'),
                    'dating_preferences': data.get('dating_preferences'),
                    'timestamp': data.get('timestamp'),
                    'summary': self._generate_analysis_summary(data.get('personality'), data.get('dating_preferences'))
                }
                
                logger.info("📊 Sending comprehensive EEG analysis results to scanner...")
                
                # Send to all connected scanner clients via WebSocket
                if self.scanner_clients:
                    disconnected = []
                    for client in self.scanner_clients:
                        try:
                            asyncio.create_task(client.send(json.dumps(results_message)))
                            logger.info("✅ Results sent to scanner client")
                        except Exception as e:
                            logger.error(f"Failed to send to scanner client: {e}")
                            disconnected.append(client)
                    
                    # Remove disconnected clients
                    for client in disconnected:
                        self.scanner_clients.discard(client)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Results sent to scanner',
                    'scanner_clients': len(self.scanner_clients)
                })
                
            except Exception as e:
                logger.error(f"❌ Error sending results to scanner: {e}")
                return jsonify({'error': f'Failed to send results: {str(e)}'}), 500
        
        @self.app.route('/eeg-status', methods=['GET'])
        def get_eeg_status():
            return jsonify({
                'eeg_connected': self.eeg_streaming,
                'clients_connected': len(self.eeg_clients),
                'hardware_port': self.params.serial_port,
                'processor_available': EEG_AVAILABLE
            })
    
    def connect_openbci_hardware(self):
        """Connect to OpenBCI hardware using BrainFlow with robust error handling"""
        if not BRAINFLOW_AVAILABLE:
            logger.error("❌ BrainFlow not available. Install brainflow package.")
            return False
        
        # Set logging level to reduce BrainFlow noise
        from brainflow.board_shim import LogLevels
        BoardShim.set_log_level(LogLevels.LEVEL_WARN)
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔌 Connecting to OpenBCI Cyton (attempt {attempt + 1}/{max_retries})...")
                
                # Create board shim
                self.board_shim = BoardShim(self.board_id, self.params)
                
                # Prepare session
                self.board_shim.prepare_session()
                
                logger.info("✅ Board prepared! Starting data stream...")
                # Start data stream
                self.board_shim.start_stream()
                
                # Wait longer for stream to stabilize (from robust_test.py)
                logger.info("⏱️  Waiting for stream to stabilize...")
                time.sleep(5)
                
                # Get channel info
                eeg_channels = BoardShim.get_eeg_channels(self.board_id)
                actual_sampling_rate = BoardShim.get_sampling_rate(self.board_id)
                
                # Check if we're actually getting data (from robust_test.py)
                initial_count = self.board_shim.get_board_data_count()
                time.sleep(1)
                current_count = self.board_shim.get_board_data_count()
                
                if current_count > initial_count:
                    logger.info(f"✅ Stream active! Data flowing at ~{current_count - initial_count} samples/sec")
                    self.eeg_streaming = True
                    self.sampling_rate = actual_sampling_rate  # Use actual sampling rate
                    
                    logger.info("✓ OpenBCI connected successfully via BrainFlow")
                    logger.info(f"   Board ID: {self.board_id}")
                    logger.info(f"   Sampling Rate: {self.sampling_rate} Hz")
                    logger.info(f"   EEG Channels: {eeg_channels}")
                    
                    return True
                else:
                    logger.warning("⚠️  No data detected, retrying...")
                    raise Exception("No data flow detected")

            except Exception as e:
                logger.error(f"❌ BrainFlow Error (attempt {attempt + 1}): {e}")
                if self.board_shim:
                    try:
                        if self.board_shim.is_prepared():
                            self.board_shim.stop_stream()
                            self.board_shim.release_session()
                    except:
                        pass
                    self.board_shim = None
                
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Waiting 3 seconds before retry...")
                    time.sleep(3)
        
        logger.error("🔄 Hardware connection failed after all retries. Check:")
        logger.error("  • OpenBCI board is powered on")
        logger.error("  • USB cable is connected")
        logger.error("  • Serial port is correct")
        logger.error("  • No other software is using the device")
        return False

    def brainflow_data_reader(self):
        """Read EEG data from OpenBCI using BrainFlow in background thread"""
        packet_count = 0
        
        # Get EEG channel indices
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        logger.info(f"EEG channels: {eeg_channels}")
        
        while self.eeg_streaming and self.board_shim:
            try:
                # Get new data from BrainFlow
                data = self.board_shim.get_current_board_data(250)  # Get up to 250 samples (1 sec)
                
                if data.shape[1] > 0:  # If we have new data
                    # Extract EEG channels (BrainFlow returns data in µV already)
                    eeg_data = data[eeg_channels, :]
                    
                    # Process each sample
                    for sample_idx in range(data.shape[1]):
                        packet_count += 1
                        
                        # Get channel values for this sample (already in µV from BrainFlow)
                        channels = []
                        for ch_idx in range(len(eeg_channels)):
                            value = eeg_data[ch_idx, sample_idx]
                            channels.append(round(value, 2))
                        
                        # Pad to 8 channels if needed
                        while len(channels) < 8:
                            channels.append(0.0)
                        channels = channels[:8]  # Ensure exactly 8 channels
                        
                        # Store recent data for frequency analysis
                        for i, channel_val in enumerate(channels):
                            self.recent_channel_data[i].append(channel_val)
                            # Keep only last 2 seconds of data
                            if len(self.recent_channel_data[i]) > self.max_recent_samples:
                                self.recent_channel_data[i] = self.recent_channel_data[i][-self.max_recent_samples:]
                        
                        # Calculate frequency bands every 25 packets (10Hz update rate)
                        frequency_bands = None
                        if packet_count % 25 == 0 and self.eeg_processor and EEG_AVAILABLE:
                            try:
                                frequency_bands = self.eeg_processor.calculate_realtime_frequency_bands(self.recent_channel_data)
                            except Exception as e:
                                logger.debug(f"Frequency analysis error: {e}")
                        
                        # Create EEG data message
                        eeg_data_msg = {
                            'type': 'eeg',
                            'timestamp': time.time(),
                            'packet_num': packet_count,
                            'channels': channels,
                            'status': 'streaming',
                            'frequency_bands': frequency_bands
                        }
                        
                        # Queue for WebSocket broadcast
                        try:
                            self.eeg_data_queue.put_nowait(json.dumps(eeg_data_msg))
                        except queue.Full:
                            pass  # Drop data if queue is full
                        
                        # Enhanced logging with signal quality assessment (from robust_test.py)
                        if packet_count % 250 == 0:
                            # Signal quality assessment
                            ch1_voltage = channels[0]
                            if abs(ch1_voltage) > 200:
                                status = "🔴 HIGH"
                            elif abs(ch1_voltage) > 100:
                                status = "🟡 MED" 
                            elif abs(ch1_voltage) > 10:
                                status = "🟢 OK"
                            else:
                                status = "🟣 LOW"
                            
                            logger.info(f"📊 BrainFlow packet #{packet_count}: Ch1={ch1_voltage:+8.1f}µV [{status}], Ch2={channels[1]:+8.1f}µV")
                            
                            if frequency_bands:
                                # Enhanced frequency band reporting with mental state
                                alpha_pct = frequency_bands.get('alpha', 0)
                                beta_pct = frequency_bands.get('beta', 0)
                                delta_pct = frequency_bands.get('delta', 0)
                                
                                # Simple mental state indicator (from robust_test.py)
                                if alpha_pct > 40:
                                    mental_state = "😌 Relaxed"
                                elif beta_pct > 40:
                                    mental_state = "🧠 Focused"
                                elif delta_pct > 50:
                                    mental_state = "😴 Drowsy"
                                else:
                                    mental_state = "🤔 Mixed"
                                
                                logger.info(f"   🧠 Freq bands: δ={delta_pct:.1f}% θ={frequency_bands.get('theta', 0):.1f}% α={alpha_pct:.1f}% β={beta_pct:.1f}%")
                                logger.info(f"   Mental state: {mental_state}")
                
                # Sleep briefly to avoid overwhelming the CPU
                time.sleep(0.01)  # 10ms sleep = max 100Hz processing
                
            except Exception as e:
                logger.error(f"BrainFlow data reading error: {e}")
                break
        
        logger.info("BrainFlow data reader thread ended")

    async def eeg_websocket_handler(self, websocket, path):
        """Handle EEG WebSocket connections from frontend"""
        self.eeg_clients.add(websocket)
        logger.info(f"EEG client connected (Total: {len(self.eeg_clients)})")

        # Send initial status
        await websocket.send(json.dumps({
            'type': 'status',
            'connected': self.eeg_streaming,
            'message': 'EEG streaming active' if self.eeg_streaming else 'EEG not connected'
        }))

        try:
            # Listen for analysis requests
            async for message in websocket:
                try:
                    request = json.loads(message)
                    await self.handle_eeg_analysis_request(websocket, request)
                except Exception as e:
                    logger.error(f"Error handling EEG message: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"EEG WebSocket error: {e}")
        finally:
            self.eeg_clients.remove(websocket)
            logger.info(f"EEG client disconnected (Remaining: {len(self.eeg_clients)})")

    async def handle_eeg_analysis_request(self, websocket, request):
        """Handle EEG analysis requests"""
        if not EEG_AVAILABLE:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'EEG analysis not available. Install numpy and scipy.'
            }))
            return

        if request.get('type') == 'analyze':
            logger.info("🧠 Processing EEG analysis request...")
            
            eeg_samples = request.get('data', [])
            if not eeg_samples:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'No EEG data provided'
                }))
                return

            try:
                # Convert to processor format
                channels_data = []
                for ch in range(8):
                    channel_samples = [sample['channels'][ch] for sample in eeg_samples 
                                     if len(sample.get('channels', [])) > ch]
                    channels_data.append(np.array(channel_samples))

                # Analyze with scientific backend
                love_analysis = self.eeg_processor.calculate_love_score(channels_data)
                frequency_analysis = self.eeg_processor.get_frequency_summary(channels_data)

                logger.info(f"✅ Analysis complete: Love Score = {love_analysis['love_score']}")

                # Send results
                await websocket.send(json.dumps({
                    'type': 'analysis',
                    'love_analysis': love_analysis,
                    'frequency_summary': frequency_analysis,
                    'method': 'scientific_backend'
                }))

            except Exception as e:
                logger.error(f"❌ EEG analysis failed: {e}")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Analysis failed: {str(e)}'
                }))

    async def broadcast_eeg_data(self):
        """Broadcast EEG data to all connected clients"""
        while True:
            if not self.eeg_data_queue.empty():
                data = self.eeg_data_queue.get()
                
                if self.eeg_clients:
                    disconnected = set()
                    for client in self.eeg_clients:
                        try:
                            await client.send(data)
                        except:
                            disconnected.add(client)
                    
                    # Remove disconnected clients
                    self.eeg_clients -= disconnected
            
            await asyncio.sleep(0.01)  # 100Hz max

    def start_eeg_server(self):
        """Start EEG WebSocket server"""
        async def run_eeg_server():
            logger.info(f"Starting EEG WebSocket server on port {self.eeg_server_port}")
            
            # Start EEG data broadcaster
            broadcast_task = asyncio.create_task(self.broadcast_eeg_data())
            
            # Create wrapper for websocket handler (newer websockets library only passes websocket)
            async def handler_wrapper(websocket):
                await self.eeg_websocket_handler(websocket, "/")
            
            # Start WebSocket server
            async with websockets.serve(handler_wrapper, "localhost", self.eeg_server_port):
                await asyncio.Future()  # Run forever
        
        # Run EEG server in separate thread
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_eeg_server())
        
        eeg_thread = threading.Thread(target=run_in_thread, daemon=True)
        eeg_thread.start()

    def start_eeg_hardware(self):
        """Start EEG hardware connection when user connects"""
        if not self.eeg_streaming:
            if self.connect_openbci_hardware():
                # Start serial reader thread
                serial_thread = threading.Thread(target=self.brainflow_data_reader, daemon=True)
                serial_thread.start()
                logger.info("EEG hardware streaming started")
            else:
                logger.error("Failed to start EEG hardware")

    def stop_eeg_hardware(self):
        """Stop EEG hardware when user disconnects"""
        if self.eeg_streaming:
            self.eeg_streaming = False
            if self.board_shim:
                try:
                    self.board_shim.stop_stream()
                    self.board_shim.release_session()
                    logger.info("✓ BrainFlow session released")
                except Exception as e:
                    logger.warning(f"Warning during BrainFlow cleanup: {e}")
                self.board_shim = None
            logger.info("EEG hardware stopped")
    
    async def connect_to_relayer(self):
        """Connect to the relayer server and register booth"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info(f"Attempting to connect to relayer at {self.relayer_url}")
                
                # SSL configuration for WSS connections
                if self.relayer_url.startswith('wss://'):
                    ssl_context = ssl.create_default_context()
                    # For self-signed certificates, disable verification
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    self.websocket = await websockets.connect(self.relayer_url, ssl=ssl_context)
                else:
                    self.websocket = await websockets.connect(self.relayer_url)
                
                # Register booth
                registration_message = {
                    "type": "register_booth",
                    "booth_id": self.booth_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.websocket.send(json.dumps(registration_message))
                logger.info(f"Booth {self.booth_id} registration sent")
                
                self.is_connected = True
                self.connection_status = "connected"
                
                # Listen for messages
                await self.listen_for_messages()
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Connection failed (attempt {retry_count}): {e}")
                self.is_connected = False
                self.connection_status = "connection_failed"
                
                if retry_count < max_retries:
                    await asyncio.sleep(5)  # Wait before retry
                else:
                    logger.error("Max retries reached. Could not connect to relayer.")
                    break
    
    async def listen_for_messages(self):
        """Listen for messages from the relayer server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(data)
                except json.JSONDecodeError:
                    logger.error("Received invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to relayer closed")
            self.is_connected = False
            self.scanner_connected = False
            self.connection_status = "disconnected"
        
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            self.is_connected = False
            self.connection_status = "error"
    
    async def handle_message(self, data):
        """Handle different types of messages from relayer"""
        message_type = data.get('type')
        
        logger.info(f"Booth received message: {data}")
        
        if message_type == 'registration_success':
            logger.info(f"Booth {self.booth_id} registered successfully")
            self.connection_status = "registered"
        
        elif message_type == 'scanner_connected':
            logger.info("Scanner connected to booth")
            self.scanner_connected = True
            self.connection_status = "user_connected"
            
            # Start EEG hardware when user connects
            logger.info("Starting EEG hardware for connected user...")
            self.start_eeg_hardware()
            
            # Send welcome message to scanner
            welcome_message = {
                "type": "relay_message",
                "data": {
                    "message": f"Welcome to Booth {self.booth_id}!",
                    "status": "ready",
                    "booth_info": {
                        "name": f"EEG Booth {self.booth_id}",
                        "capabilities": ["eeg_recording", "brain_analysis"]
                    }
                }
            }
            await self.send_to_relayer(welcome_message)
        
        elif message_type == 'scanner_disconnected':
            logger.info("Scanner disconnected from booth")
            self.scanner_connected = False
            self.connection_status = "registered"
            
            # Stop EEG hardware when user disconnects
            logger.info("Stopping EEG hardware...")
            self.stop_eeg_hardware()
        
        elif message_type == 'message_from_scanner':
            scanner_data = data.get('data', {})
            logger.info(f"Message from scanner: {scanner_data}")
            
            # Handle different scanner messages
            if scanner_data.get('action') == 'connection_established':
                logger.info("Scanner established connection")
                self.connection_status = "user_connected"
                
                # Respond to scanner
                response = {
                    "type": "relay_message",
                    "data": {
                        "message": "Connection confirmed",
                        "status": "booth_ready",
                        "next_steps": "Please wait for EEG setup instructions"
                    }
                }
                await self.send_to_relayer(response)
            
            elif scanner_data.get('action') == 'start_session':
                logger.info("Starting EEG session")
                # Simulate EEG session start
                response = {
                    "type": "relay_message",
                    "data": {
                        "message": "EEG session started",
                        "status": "recording",
                        "session_id": f"session_{uuid.uuid4().hex[:8]}"
                    }
                }
                await self.send_to_relayer(response)
        
        elif message_type == 'error':
            logger.error(f"Error from relayer: {data.get('message')}")
            self.connection_status = "error"
    
    async def send_to_relayer(self, message):
        """Send message to relayer server"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.send(json.dumps(message))
                logger.info(f"Sent to relayer: {message}")
            except Exception as e:
                logger.error(f"Failed to send message to relayer: {e}")
        else:
            logger.warning("Cannot send message: not connected to relayer")
    
    def start_flask_server(self):
        """Start Flask server in a separate thread"""
        def run_flask():
            self.app.run(host='0.0.0.0', port=self.frontend_port, debug=False)
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info(f"Flask server started on port {self.frontend_port}")
    
    def _generate_analysis_summary(self, personality, dating_preferences):
        """Generate a human-readable summary of the EEG analysis"""
        if not personality or not dating_preferences:
            return "Incomplete analysis data"
        
        # Find dominant personality traits (>70%)
        dominant_traits = []
        for trait, score in personality.get('scores', {}).items():
            if score >= 70:
                dominant_traits.append(trait.capitalize())
        
        # Find top dating preferences
        sorted_prefs = sorted(dating_preferences, key=lambda x: x.get('attraction_score', 0), reverse=True)
        top_preferences = []
        
        for pref in sorted_prefs[:3]:  # Top 3 preferences
            if pref.get('attraction_score', 0) >= 60:
                top_preferences.append({
                    'type': pref.get('stimulus_type', '').replace('_', ' ').title(),
                    'score': pref.get('attraction_score', 0),
                    'level': pref.get('attraction_level', '')
                })
        
        summary = {
            'personality_highlights': dominant_traits if dominant_traits else ['Balanced personality'],
            'top_preferences': top_preferences,
            'analysis_confidence': personality.get('confidence', 0),
            'total_preferences_tested': len(dating_preferences),
            'interpretation': self._generate_interpretation(personality, top_preferences)
        }
        
        return summary
    
    def _generate_interpretation(self, personality, top_preferences):
        """Generate personality-based interpretation"""
        scores = personality.get('scores', {})
        
        # Generate interpretation based on personality
        interpretation = []
        
        if scores.get('extraversion', 50) >= 70:
            interpretation.append("Shows strong social engagement patterns")
        elif scores.get('extraversion', 50) <= 30:
            interpretation.append("Prefers deeper, meaningful connections")
        
        if scores.get('openness', 50) >= 70:
            interpretation.append("Values creativity and novel experiences")
        
        if top_preferences:
            interpretation.append(f"Shows clear attraction preferences with {top_preferences[0]['level'].lower()}")
        
        return ' • '.join(interpretation) if interpretation else "Balanced neural response patterns detected"

    async def run(self):
        """Main run method"""
        logger.info(f"Starting Booth Backend with ID: {self.booth_id}")
        
        # Start Flask server for frontend communication
        self.start_flask_server()
        
        # Start EEG WebSocket server
        self.start_eeg_server()
        logger.info(f"EEG WebSocket server available at ws://localhost:{self.eeg_server_port}")
        
        # Connect to relayer server
        await self.connect_to_relayer()

async def main():
    # You can specify booth_id as command line argument or let it auto-generate
    import sys
    
    booth_id = None
    if len(sys.argv) > 1:
        booth_id = sys.argv[1]
    
    booth = BoothBackend(booth_id=booth_id)
    
    try:
        await booth.run()
    except KeyboardInterrupt:
        logger.info("Booth backend stopped by user")
    except Exception as e:
        logger.error(f"Booth backend error: {e}")

if __name__ == "__main__":
    asyncio.run(main())