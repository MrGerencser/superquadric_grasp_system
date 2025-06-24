import os
import cv2
import numpy as np
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional, Any

class WebInterface:
    """Manages web-based visualization interface - improved version"""
    
    def __init__(self, web_dir: str = "/tmp/grasp_system_live", port: int = 8080):
        self.web_dir = web_dir
        self.port = port
        self.web_server_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps_values = []
        self.last_frame_time = time.time()
        self.is_running = False
        
        # High-frequency update tracking
        self.last_web_update = 0
        self.web_update_interval = 0.066  # ~15 FPS (66ms)
        
    def initialize(self) -> bool:
        """Initialize web interface"""
        try:
            # Create web directory
            os.makedirs(self.web_dir, exist_ok=True)
            
            # Create initial placeholder image
            self._create_placeholder_image()
            
            # Create improved HTML interface
            self._create_html_interface()
            
            # Start web server
            self._start_web_server()
            
            self.is_running = True
            print(f"LIVE web interface available at: http://localhost:{self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize web interface: {e}")
            return False
    
    def update_frame(self, frame: np.ndarray, quality: int = 85):
        """Update the displayed frame with high-frequency support"""
        try:
            current_time = time.time()
            
            # Throttle updates to maintain smooth ~15 FPS display
            if current_time - self.last_web_update < self.web_update_interval:
                return  # Skip this update to maintain timing
            
            with self.frame_lock:
                self.latest_frame = frame.copy()
                self.frame_count += 1
                
                # Use atomic write (temp file then rename)
                web_image_path = f"{self.web_dir}/latest.jpg"
                temp_path = f"{self.web_dir}/latest_temp.jpg"
                
                # Write to temporary file
                success = cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                if success:
                    # Atomic rename prevents reading partial files
                    if os.path.exists(temp_path):
                        os.rename(temp_path, web_image_path)
                
                # Update timing
                self.last_web_update = current_time
                self._update_fps()
                
        except Exception as e:
            print(f"Error updating web frame: {e}")
    
    def _create_placeholder_image(self):
        """Create initial placeholder image"""
        try:
            # Create a more informative placeholder
            placeholder = np.zeros((400, 1280, 3), dtype=np.uint8)
            placeholder[:, :] = [30, 30, 50]  # Dark blue background
            
            # Add title
            cv2.putText(placeholder, "Superquadric Grasp System", 
                       (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Add status
            cv2.putText(placeholder, "Initializing Live Feed...", 
                       (450, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(placeholder, f"Started: {timestamp}", 
                       (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Save placeholder
            placeholder_path = f"{self.web_dir}/latest.jpg"
            cv2.imwrite(placeholder_path, placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            print(f"Created placeholder image at: {placeholder_path}")
            
        except Exception as e:
            print(f"Error creating placeholder image: {e}")
    
    def _create_html_interface(self):
        """Create improved HTML interface matching the working version"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>LIVE Superquadric Grasp System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .live-indicator {
            text-align: center;
            background-color: #28a745;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        #live_image {
            max-width: 100%;
            height: auto;
            border: 2px solid #28a745;
            border-radius: 5px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 16px;
        }
        button:hover {
            background-color: #218838;
        }
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status.connected {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.disconnected {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .fps-display {
            font-weight: bold;
            color: #28a745;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LIVE Superquadric Grasp System</h1>
        
        <div class="live-indicator" id="live-indicator">
            LIVE DETECTION FEED - Real-time Processing
        </div>
        
        <div id="status" class="status connected">
            Live feed active - High frequency updates
        </div>
        
        <div class="controls">
            <button id="toggle-btn" onclick="toggleRefresh()">Pause</button>
            <button onclick="saveImage()">Save Frame</button>
            <button onclick="toggleFullscreen()">Fullscreen</button>
            <button onclick="resetConnection()">Reset</button>
        </div>
        
        <div class="image-container">
            <img id="live_image" src="latest.jpg" alt="Loading live feed..." />
        </div>
        
        <div class="fps-display">
            Display FPS: <span id="fps">--</span> | 
            Last updated: <span id="timestamp">--</span> |
            Total frames: <span id="frame_count">0</span>
        </div>
    </div>

    <script>
        let refreshInterval;
        let isRefreshing = true;
        let frameCount = 0;
        let lastFrameTime = Date.now();
        let errorCount = 0;
        
        function updateImage() {
            if (!isRefreshing) return;
            
            const img = document.getElementById('live_image');
            const timestamp = Date.now();
            
            // Create new image to test loading
            const testImg = new Image();
            
            testImg.onload = function() {
                // Calculate display FPS
                const timeDiff = timestamp - lastFrameTime;
                const displayFPS = timeDiff > 0 ? (1000 / timeDiff).toFixed(1) : 0;
                
                // Update image with cache busting
                img.src = 'latest.jpg?' + timestamp + '&r=' + Math.random();
                
                // Update displays
                document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();
                document.getElementById('fps').textContent = displayFPS;
                document.getElementById('frame_count').textContent = frameCount++;
                
                // Update status
                const status = document.getElementById('status');
                status.textContent = 'Live feed active - High frequency updates';
                status.className = 'status connected';
                
                lastFrameTime = timestamp;
                errorCount = 0;
            };
            
            testImg.onerror = function() {
                errorCount++;
                if (errorCount > 3) {
                    const status = document.getElementById('status');
                    status.textContent = 'Connection lost - Check if system is running';
                    status.className = 'status disconnected';
                }
            };
            
            testImg.src = 'latest.jpg?' + timestamp + '&r=' + Math.random();
        }
        
        function toggleRefresh() {
            isRefreshing = !isRefreshing;
            const button = document.getElementById('toggle-btn');
            const indicator = document.getElementById('live-indicator');
            const status = document.getElementById('status');
            
            if (isRefreshing) {
                button.textContent = 'Pause';
                indicator.textContent = 'LIVE DETECTION FEED - Real-time Processing';
                indicator.style.backgroundColor = '#28a745';
                status.textContent = 'Live feed active - High frequency updates';
                status.className = 'status connected';
            } else {
                button.textContent = 'Resume';
                indicator.textContent = 'PAUSED - Click Resume to Continue';
                indicator.style.backgroundColor = '#ffc107';
                status.textContent = 'Paused - Click resume to continue';
                status.className = 'status disconnected';
            }
        }
        
        function saveImage() {
            const link = document.createElement('a');
            link.href = 'latest.jpg?' + Date.now();
            link.download = 'grasp_system_' + new Date().toISOString().slice(0,19).replace(/:/g, '-') + '.jpg';
            link.click();
        }
        
        function toggleFullscreen() {
            const img = document.getElementById('live_image');
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else {
                img.requestFullscreen().catch(err => {
                    console.log('Fullscreen not supported:', err);
                });
            }
        }
        
        function resetConnection() {
            frameCount = 0;
            errorCount = 0;
            document.getElementById('frame_count').textContent = '0';
            updateImage();
        }
        
        // HIGH FREQUENCY: Match the working implementation (66ms = ~15 FPS)
        refreshInterval = setInterval(updateImage, 66);
        
        // Initial load
        updateImage();
        
        console.log('High-frequency web interface started at ~15 FPS');
    </script>
</body>
</html>
        """
        
        with open(f"{self.web_dir}/index.html", "w", encoding='utf-8') as f:
            f.write(html_content)
    
    def _start_web_server(self):
        """Start web server"""
        self.web_server_thread = threading.Thread(target=self._run_web_server, daemon=True)
        self.web_server_thread.start()
    
    def _run_web_server(self):
        """Run HTTP server"""
        try:
            web_directory = self.web_dir
            
            class Handler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=web_directory, **kwargs)
                
                def log_message(self, format, *args):
                    pass  # Suppress logs
                    
                def do_GET(self):
                    try:
                        super().do_GET()
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # Normal browser behavior
                    except Exception as e:
                        print(f"Web server request error: {e}")
            
            httpd = HTTPServer(("localhost", self.port), Handler)
            print(f"Web server started on port {self.port}")
            httpd.serve_forever()
            
        except Exception as e:
            print(f"Web server error: {e}")
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_values.append(fps)
            
            if len(self.fps_values) > 10:
                self.fps_values.pop(0)
        
        self.last_frame_time = current_time
    
    def get_average_fps(self) -> float:
        """Get average FPS"""
        if not self.fps_values:
            return 0.0
        return sum(self.fps_values) / len(self.fps_values)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.is_running = False
            print("Web interface cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")