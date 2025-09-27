import asyncio
import websockets
import json
import logging
import uuid
import threading
import ssl
import os
import serial
import time
import queue
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
try:
    import numpy as np
    from eeg_processor import EEGProcessor
    EEG_AVAILABLE = True
except ImportError:
    print("Warning: EEG processing not available. Install numpy and scipy for full functionality.")
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
        
        # OpenBCI hardware connection
        self.openbci_port = "/dev/cu.usbserial-DM01MV82"
        self.openbci_baudrate = 115200
        self.openbci_scale = 0.02235 / 1000  # Correct scale for μV
        self.eeg_serial = None
        self.eeg_streaming = False
        self.eeg_data_queue = queue.Queue(maxsize=1000)
        
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
                return jsonify({'error': 'Insufficient EEG data for analysis'}), 400
            
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
        
        @self.app.route('/eeg-status', methods=['GET'])
        def get_eeg_status():
            return jsonify({
                'eeg_connected': self.eeg_streaming,
                'clients_connected': len(self.eeg_clients),
                'hardware_port': self.openbci_port,
                'processor_available': EEG_AVAILABLE
            })
    
    def connect_openbci_hardware(self):
        """Connect to OpenBCI hardware"""
        try:
            logger.info("Connecting to OpenBCI hardware...")
            self.eeg_serial = serial.Serial(self.openbci_port, self.openbci_baudrate, timeout=0.1)
            time.sleep(2)

            # Initialize OpenBCI
            self.eeg_serial.write(b's')  # stop
            time.sleep(0.5)
            self.eeg_serial.reset_input_buffer()
            self.eeg_serial.write(b'd')  # defaults
            time.sleep(0.5)
            self.eeg_serial.write(b'b')  # begin streaming

            self.eeg_streaming = True
            logger.info("✓ OpenBCI connected successfully")
            return True

        except Exception as e:
            logger.error(f"❌ OpenBCI connection failed: {e}")
            return False

    def openbci_serial_reader(self):
        """Read EEG data from OpenBCI in background thread"""
        buffer = bytearray()
        packet_count = 0

        while self.eeg_streaming:
            try:
                if self.eeg_serial and self.eeg_serial.in_waiting:
                    buffer.extend(self.eeg_serial.read(self.eeg_serial.in_waiting))

                    while len(buffer) >= 33:
                        # Find packet start (0xA0)
                        start = buffer.index(0xA0) if 0xA0 in buffer else -1

                        if start >= 0 and start + 32 < len(buffer):
                            # Check for packet end (0xC0)
                            if buffer[start + 32] == 0xC0:
                                packet = buffer[start:start + 33]
                                packet_count += 1

                                # Parse 8 channels
                                channels = []
                                for i in range(8):
                                    idx = 2 + (i * 3)
                                    val = (packet[idx] << 16) | (packet[idx+1] << 8) | packet[idx+2]
                                    if val & 0x800000:
                                        val -= 0x1000000
                                    channels.append(round(val * self.openbci_scale, 2))

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
                                eeg_data = {
                                    'type': 'eeg',
                                    'timestamp': time.time(),
                                    'packet_num': packet_count,
                                    'channels': channels,
                                    'status': 'streaming',
                                    'frequency_bands': frequency_bands
                                }

                                # Queue for WebSocket broadcast
                                try:
                                    self.eeg_data_queue.put_nowait(json.dumps(eeg_data))
                                except queue.Full:
                                    pass  # Drop data if queue is full

                                # Log every 50 packets
                                if packet_count % 50 == 0:
                                    logger.info(f"EEG packet #{packet_count}: Ch1={channels[0]:.2f}μV")

                                buffer = buffer[start + 33:]
                            else:
                                buffer = buffer[start + 1:]
                        else:
                            if len(buffer) > 100:
                                buffer = buffer[-33:]
                            break

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"EEG serial error: {e}")
                break

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
                serial_thread = threading.Thread(target=self.openbci_serial_reader, daemon=True)
                serial_thread.start()
                logger.info("EEG hardware streaming started")
            else:
                logger.error("Failed to start EEG hardware")

    def stop_eeg_hardware(self):
        """Stop EEG hardware when user disconnects"""
        if self.eeg_streaming:
            self.eeg_streaming = False
            if self.eeg_serial:
                try:
                    self.eeg_serial.close()
                except:
                    pass
                self.eeg_serial = None
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