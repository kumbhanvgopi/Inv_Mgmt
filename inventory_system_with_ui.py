# ==================== AUTONOMOUS INVENTORY OPTIMIZATION SYSTEM WITH WEB UI ====================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import queue
import time
import warnings
warnings.filterwarnings('ignore')

# Web server imports
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import webbrowser
from threading import Timer

# External libraries for ML
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy import stats
    from scipy.optimize import minimize
except ImportError:
    print("Note: Some ML libraries not available in this environment")

# ==================== DATA STRUCTURES ====================

@dataclass
class Product:
    sku: str
    name: str
    category: str
    price: float
    lead_time: int
    supplier_id: str
    abc_class: str = "C"
    xyz_class: str = "Z"

@dataclass
class InventoryData:
    sku: str
    location: str
    current_stock: int
    reserved_stock: int
    available_stock: int
    last_updated: datetime

@dataclass
class DemandForecast:
    sku: str
    location: str
    forecast_horizon: int
    predicted_demand: List[float]
    confidence_interval: Tuple[float, float]
    accuracy_score: float
    timestamp: datetime

@dataclass
class ReorderRecommendation:
    sku: str
    location: str
    current_stock: int
    reorder_point: float
    recommended_quantity: int
    urgency_level: str
    reasoning: str
    timestamp: datetime

@dataclass
class Message:
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1

class MessageType(Enum):
    DEMAND_UPDATE = "demand_update"
    INVENTORY_STATUS = "inventory_status"
    REORDER_ALERT = "reorder_alert"
    ALLOCATION_REQUEST = "allocation_request"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"
    ABC_UPDATE = "abc_update"
    SAFETY_STOCK_UPDATE = "safety_stock_update"

# ==================== BASE AGENT ARCHITECTURE ====================

class BaseAgent(ABC):
    def __init__(self, agent_id: str, system_manager):
        self.agent_id = agent_id
        self.system_manager = system_manager
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.performance_metrics = {}
        
    def start(self):
        """Start the agent's main processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def send_message(self, receiver: str, message_type: str, content: Dict[str, Any], priority: int = 1):
        """Send message to another agent"""
        message = Message(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority
        )
        self.system_manager.route_message(message)
    
    def receive_message(self, message: Message):
        """Receive message from another agent"""
        self.message_queue.put(message)
    
    def _run(self):
        """Main agent processing loop"""
        while self.running:
            try:
                # Process messages
                while not self.message_queue.empty():
                    try:
                        message = self.message_queue.get_nowait()
                        self.handle_message(message)
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Error processing message in agent {self.agent_id}: {e}")
                
                # Perform main agent tasks
                try:
                    self.execute_main_task()
                except Exception as e:
                    print(f"Error in main task for agent {self.agent_id}: {e}")
                
                # Sleep to prevent excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                print(f"Critical error in agent {self.agent_id}: {e}")
                time.sleep(5)
    
    @abstractmethod
    def handle_message(self, message: Message):
        """Handle incoming messages"""
        pass
    
    @abstractmethod
    def execute_main_task(self):
        """Execute main agent functionality"""
        pass

# ==================== DEMAND FORECASTING AGENT ====================

class DemandForecastingAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.models = {}
        self.historical_data = {}
        self.forecast_cache = {}
        self.scaler = StandardScaler()
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.DEMAND_UPDATE.value:
            self.update_historical_data(message.content)
        elif message.message_type == MessageType.SEASONAL_ADJUSTMENT.value:
            self.apply_seasonal_adjustments(message.content)
    
    def execute_main_task(self):
        """Generate demand forecasts for all products"""
        current_time = datetime.now()
        
        # Update forecasts every hour
        if not hasattr(self, 'last_forecast_time') or \
           (current_time - self.last_forecast_time).seconds > 300:  # Reduced to 5 minutes for demo
            
            self.generate_forecasts()
            self.last_forecast_time = current_time
    
    def update_historical_data(self, data: Dict[str, Any]):
        """Update historical demand data"""
        sku = data['sku']
        if sku not in self.historical_data:
            self.historical_data[sku] = []
        
        self.historical_data[sku].append({
            'date': data['date'],
            'demand': data['demand'],
            'price': data.get('price', 0),
            'promotion': data.get('promotion', False),
            'weather': data.get('weather', 'normal'),
            'season': self._get_season(data['date'])
        })
    
    def _get_season(self, date_str: str) -> str:
        """Determine season from date"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        except:
            return 'unknown'
    
    def prepare_features(self, sku: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for machine learning model"""
        if sku not in self.historical_data or len(self.historical_data[sku]) < 10:
            return None, None
        
        data = pd.DataFrame(self.historical_data[sku])
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Create features
        features = []
        targets = []
        
        for i in range(7, len(data)):
            # Temporal features
            current_date = data.iloc[i]['date']
            day_of_week = current_date.weekday()
            day_of_month = current_date.day
            month = current_date.month
            
            # Historical demand features
            recent_demand = data.iloc[i-7:i]['demand'].values
            avg_demand_7d = np.mean(recent_demand)
            std_demand_7d = np.std(recent_demand)
            trend = recent_demand[-1] - recent_demand[0]
            
            # External factors
            price = data.iloc[i]['price']
            promotion = 1 if data.iloc[i]['promotion'] else 0
            season_encoded = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}.get(
                data.iloc[i]['season'], 0)
            
            feature_vector = [
                day_of_week, day_of_month, month,
                avg_demand_7d, std_demand_7d, trend,
                price, promotion, season_encoded
            ] + recent_demand.tolist()
            
            features.append(feature_vector)
            targets.append(data.iloc[i]['demand'])
        
        return np.array(features), np.array(targets)
    
    def train_model(self, sku: str):
        """Train forecasting model for specific SKU"""
        X, y = self.prepare_features(sku)
        if X is None or len(X) < 20:
            return
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        self.models[sku] = {
            'model': model,
            'scaler': self.scaler,
            'mae': mae,
            'mse': mse,
            'last_trained': datetime.now()
        }
    
    def generate_forecast(self, sku: str, horizon: int = 14) -> Optional[DemandForecast]:
        """Generate demand forecast for specific SKU"""
        if sku not in self.models:
            self.train_model(sku)
        
        if sku not in self.models:
            return None
        
        model_info = self.models[sku]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Get recent data for prediction
        if sku not in self.historical_data:
            return None
        
        recent_data = self.historical_data[sku][-7:]
        if len(recent_data) < 7:
            return None
        
        forecasts = []
        current_data = recent_data.copy()
        
        for day in range(horizon):
            # Prepare features for prediction
            future_date = datetime.now() + timedelta(days=day+1)
            day_of_week = future_date.weekday()
            day_of_month = future_date.day
            month = future_date.month
            
            recent_demand = [d['demand'] for d in current_data[-7:]]
            avg_demand_7d = np.mean(recent_demand)
            std_demand_7d = np.std(recent_demand)
            trend = recent_demand[-1] - recent_demand[0]
            
            # Assume current price and no promotion for future
            price = current_data[-1]['price']
            promotion = 0
            season_encoded = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}.get(
                self._get_season(future_date.strftime('%Y-%m-%d')), 0)
            
            feature_vector = np.array([[
                day_of_week, day_of_month, month,
                avg_demand_7d, std_demand_7d, trend,
                price, promotion, season_encoded
            ] + recent_demand])
            
            # Make prediction
            feature_scaled = scaler.transform(feature_vector)
            prediction = model.predict(feature_scaled)[0]
            prediction = max(0, prediction)
            
            forecasts.append(prediction)
            
            # Update current_data for next iteration
            current_data.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'demand': prediction,
                'price': price,
                'promotion': False,
                'season': self._get_season(future_date.strftime('%Y-%m-%d'))
            })
        
        # Calculate confidence intervals (simplified)
        mae = model_info['mae']
        confidence_lower = max(0, np.mean(forecasts) - 1.96 * mae)
        confidence_upper = np.mean(forecasts) + 1.96 * mae
        
        forecast = DemandForecast(
            sku=sku,
            location="main",
            forecast_horizon=horizon,
            predicted_demand=forecasts,
            confidence_interval=(confidence_lower, confidence_upper),
            accuracy_score=1 / (1 + mae),
            timestamp=datetime.now()
        )
        
        self.forecast_cache[sku] = forecast
        return forecast
    
    def generate_forecasts(self):
        """Generate forecasts for all products and broadcast updates"""
        for sku in self.historical_data.keys():
            forecast = self.generate_forecast(sku)
            if forecast:
                # Broadcast forecast to other agents
                self.send_message(
                    receiver="all",
                    message_type=MessageType.DEMAND_UPDATE.value,
                    content={
                        "sku": sku,
                        "forecast": {
                            "predicted_demand": forecast.predicted_demand,
                            "confidence_interval": forecast.confidence_interval,
                            "timestamp": forecast.timestamp.isoformat()
                        }
                    }
                )
    
    def apply_seasonal_adjustments(self, adjustment_data: Dict[str, Any]):
        """Apply seasonal adjustments to forecasts"""
        sku = adjustment_data['sku']
        seasonal_factor = adjustment_data['seasonal_factor']
        
        if sku in self.forecast_cache:
            forecast = self.forecast_cache[sku]
            adjusted_demand = [d * seasonal_factor for d in forecast.predicted_demand]
            forecast.predicted_demand = adjusted_demand
            self.forecast_cache[sku] = forecast

# ==================== SIMPLIFIED OTHER AGENTS ====================
# (Including other agents from original code - abbreviated for space)

class StockLevelMonitoringAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.inventory_data = {}
        self.alert_thresholds = {}
        
    def handle_message(self, message: Message):
        try:
            if message.message_type == MessageType.INVENTORY_STATUS.value:
                self.update_inventory_data(message.content)
            elif message.message_type == MessageType.REORDER_ALERT.value:
                self.process_reorder_alert(message.content)
        except Exception as e:
            print(f"Error handling message in {self.agent_id}: {e}")
    
    def execute_main_task(self):
        """Monitor stock levels and detect anomalies"""
        try:
            self.check_stock_levels()
            self.detect_anomalies()
            self.broadcast_status_updates()
        except Exception as e:
            print(f"Error in main task for {self.agent_id}: {e}")
    
    def update_inventory_data(self, data: Dict[str, Any]):
        """Update inventory data from external systems"""
        sku = data.get('sku', 'UNKNOWN')
        location = data.get('location', 'main')
        current_stock = data.get('current_stock', 0)
        reserved_stock = data.get('reserved_stock', 0)
        
        key = f"{sku}_{location}"
        self.inventory_data[key] = InventoryData(
            sku=sku,
            location=location,
            current_stock=current_stock,
            reserved_stock=reserved_stock,
            available_stock=current_stock - reserved_stock,
            last_updated=datetime.now()
        )
    
    def check_stock_levels(self):
        """Check current stock levels against thresholds"""
        for key, inventory in self.inventory_data.items():
            try:
                sku = inventory.sku
                location = getattr(inventory, 'location', 'main')
                
                if f"{sku}_reorder_point" in self.alert_thresholds:
                    reorder_point = self.alert_thresholds[f"{sku}_reorder_point"]
                    
                    if inventory.available_stock <= reorder_point:
                        self.send_message(
                            receiver="reorder_point_agent",
                            message_type=MessageType.REORDER_ALERT.value,
                            content={
                                "sku": sku,
                                "location": location,
                                "current_stock": inventory.current_stock,
                                "available_stock": inventory.available_stock,
                                "reorder_point": reorder_point,
                                "urgency": "high" if inventory.available_stock <= reorder_point * 0.5 else "medium"
                            },
                            priority=2
                        )
            except Exception as e:
                print(f"Error checking stock levels for {key}: {e}")
                continue
    
    def detect_anomalies(self):
        """Detect anomalies in stock movement patterns"""
        for key, inventory in self.inventory_data.items():
            try:
                location = getattr(inventory, 'location', 'main')
                
                if hasattr(self, 'previous_inventory_data') and \
                   key in self.previous_inventory_data:
                    
                    prev_stock = self.previous_inventory_data[key].current_stock
                    current_stock = inventory.current_stock
                    stock_change = prev_stock - current_stock
                    
                    if abs(stock_change) > 100:
                        self.send_message(
                            receiver="all",
                            message_type=MessageType.INVENTORY_STATUS.value,
                            content={
                                "type": "anomaly_detected",
                                "sku": inventory.sku,
                                "location": location,
                                "stock_change": stock_change,
                                "timestamp": datetime.now().isoformat()
                            },
                            priority=3
                        )
            except Exception as e:
                print(f"Error detecting anomalies for {key}: {e}")
                continue
        
        self.previous_inventory_data = self.inventory_data.copy()
    
    def broadcast_status_updates(self):
        """Broadcast inventory status to other agents"""
        for key, inventory in self.inventory_data.items():
            try:
                location = getattr(inventory, 'location', 'main')
                
                self.send_message(
                    receiver="all",
                    message_type=MessageType.INVENTORY_STATUS.value,
                    content={
                        "type": "status_update",
                        "sku": inventory.sku,
                        "location": location,
                        "current_stock": inventory.current_stock,
                        "available_stock": inventory.available_stock,
                        "last_updated": inventory.last_updated.isoformat()
                    }
                )
            except Exception as e:
                print(f"Error broadcasting status for {key}: {e}")
                continue
    
    def process_reorder_alert(self, alert_data: Dict[str, Any]):
        """Process reorder alerts from other agents"""
        try:
            sku = alert_data.get('sku', 'UNKNOWN')
            self.alert_thresholds[f"{sku}_last_alert"] = datetime.now()
        except Exception as e:
            print(f"Error processing reorder alert: {e}")

class ReorderPointAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.reorder_points = {}
        self.demand_forecasts = {}
        self.lead_times = {}
        self.safety_stocks = {}
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.DEMAND_UPDATE.value:
            self.update_demand_forecast(message.content)
        elif message.message_type == MessageType.SAFETY_STOCK_UPDATE.value:
            self.update_safety_stock(message.content)
    
    def execute_main_task(self):
        """Calculate and update reorder points"""
        self.calculate_reorder_points()
    
    def update_demand_forecast(self, forecast_data: Dict[str, Any]):
        """Update demand forecast data"""
        sku = forecast_data['sku']
        self.demand_forecasts[sku] = forecast_data['forecast']
    
    def update_safety_stock(self, safety_stock_data: Dict[str, Any]):
        """Update safety stock levels"""
        sku = safety_stock_data['sku']
        self.safety_stocks[sku] = safety_stock_data['safety_stock']
    
    def calculate_reorder_point(self, sku: str) -> float:
        """Calculate reorder point for specific SKU"""
        if sku not in self.demand_forecasts:
            return 0
        
        forecast = self.demand_forecasts[sku]
        predicted_demand = forecast.get('predicted_demand', [])
        
        if not predicted_demand:
            return 0
        
        avg_daily_demand = np.mean(predicted_demand[:7])
        lead_time = self.lead_times.get(sku, 7)
        safety_stock = self.safety_stocks.get(sku, avg_daily_demand * 2)
        
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        
        return max(0, reorder_point)
    
    def calculate_reorder_points(self):
        """Calculate reorder points for all SKUs"""
        for sku in self.demand_forecasts.keys():
            reorder_point = self.calculate_reorder_point(sku)
            self.reorder_points[sku] = reorder_point
            
            self.send_message(
                receiver="stock_monitoring_agent",
                message_type=MessageType.INVENTORY_STATUS.value,
                content={
                    "type": "reorder_point_update",
                    "sku": sku,
                    "reorder_point": reorder_point
                }
            )

# ==================== SIMPLIFIED ADDITIONAL AGENTS ====================

class InventoryAllocationAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.pending_orders = {}
        
    def handle_message(self, message: Message):
        pass
    
    def execute_main_task(self):
        pass

class SeasonalAdjustmentAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.adjustment_factors = {}
        
    def handle_message(self, message: Message):
        pass
    
    def execute_main_task(self):
        self.update_seasonal_factors()
        self.broadcast_seasonal_adjustments()
    
    def update_seasonal_factors(self):
        """Update seasonal adjustment factors"""
        current_month = datetime.now().month
        base_seasonal_factors = {
            1: 0.8, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
            7: 1.15, 8: 1.1, 9: 0.95, 10: 1.0, 11: 1.2, 12: 1.4
        }
        
        # Simulate some products for seasonal adjustment
        sample_skus = ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']
        for sku in sample_skus:
            base_factor = base_seasonal_factors.get(current_month, 1.0)
            self.adjustment_factors[sku] = base_factor
    
    def broadcast_seasonal_adjustments(self):
        """Broadcast seasonal adjustments to other agents"""
        for sku, factor in self.adjustment_factors.items():
            self.send_message(
                receiver="demand_forecasting_agent",
                message_type=MessageType.SEASONAL_ADJUSTMENT.value,
                content={
                    "sku": sku,
                    "seasonal_factor": factor,
                    "month": datetime.now().month,
                    "timestamp": datetime.now().isoformat()
                }
            )

class ABCClassificationAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.classifications = {}
        
    def handle_message(self, message: Message):
        pass
    
    def execute_main_task(self):
        # Simulate ABC classification
        sample_skus = ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']
        abc_classes = ['A', 'B', 'A', 'C', 'B']
        xyz_classes = ['X', 'Y', 'X', 'Z', 'Y']
        
        for i, sku in enumerate(sample_skus):
            self.classifications[sku] = {
                'abc_class': abc_classes[i],
                'xyz_class': xyz_classes[i]
            }
            
            self.send_message(
                receiver="all",
                message_type=MessageType.ABC_UPDATE.value,
                content={
                    "sku": sku,
                    "abc_class": abc_classes[i],
                    "xyz_class": xyz_classes[i],
                    "timestamp": datetime.now().isoformat()
                }
            )

class SafetyStockOptimizationAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.safety_stocks = {}
        self.service_levels = {}
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.ABC_UPDATE.value:
            self.update_classification_data(message.content)
    
    def execute_main_task(self):
        self.calculate_optimal_safety_stocks()
        self.broadcast_safety_stock_updates()
    
    def update_classification_data(self, data: Dict[str, Any]):
        """Update product classification data"""
        sku = data['sku']
        abc_class = data['abc_class']
        
        service_level_map = {
            'A': 0.98,
            'B': 0.95,
            'C': 0.90
        }
        
        self.service_levels[sku] = service_level_map.get(abc_class, 0.90)
    
    def calculate_optimal_safety_stocks(self):
        """Calculate optimal safety stock levels for all products"""
        sample_skus = ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']
        for sku in sample_skus:
            # Simplified safety stock calculation
            service_level = self.service_levels.get(sku, 0.95)
            z_score = stats.norm.ppf(service_level) if 'stats' in globals() else 1.64
            safety_stock = z_score * 10 * np.sqrt(7)  # Simplified calculation
            
            self.safety_stocks[sku] = max(5, safety_stock)
    
    def broadcast_safety_stock_updates(self):
        """Broadcast safety stock updates to other agents"""
        for sku, safety_stock in self.safety_stocks.items():
            self.send_message(
                receiver="reorder_point_agent",
                message_type=MessageType.SAFETY_STOCK_UPDATE.value,
                content={
                    "sku": sku,
                    "safety_stock": safety_stock,
                    "service_level": self.service_levels.get(sku, 0.95),
                    "timestamp": datetime.now().isoformat()
                }
            )

# ==================== SYSTEM MANAGER WITH WEB SERVER ====================

class InventoryOptimizationSystem:
    def __init__(self):
        self.agents = {}
        self.message_router = {}
        self.running = False
        self.products = {}
        self.performance_metrics = {}
        self.web_app = Flask(__name__)
        CORS(self.web_app)
        self.setup_web_routes()
        
    def setup_web_routes(self):
        """Setup Flask web routes"""
        
        @self.web_app.route('/')
        def dashboard():
            return render_template_string(HTML_TEMPLATE)
        
        @self.web_app.route('/api/status')
        def api_status():
            return jsonify(self.get_system_status())
        
        @self.web_app.route('/api/metrics')
        def api_metrics():
            return jsonify(self.get_performance_metrics())
        
        @self.web_app.route('/api/forecasts')
        def api_forecasts():
            demand_agent = self.agents.get("demand_forecasting_agent")
            if demand_agent and hasattr(demand_agent, 'forecast_cache'):
                forecasts = {}
                for sku, forecast in demand_agent.forecast_cache.items():
                    forecasts[sku] = {
                        'predicted_demand': forecast.predicted_demand[:7],  # Next 7 days
                        'confidence_interval': forecast.confidence_interval,
                        'accuracy_score': forecast.accuracy_score,
                        'timestamp': forecast.timestamp.isoformat()
                    }
                return jsonify(forecasts)
            return jsonify({})
        
        @self.web_app.route('/api/inventory')
        def api_inventory():
            stock_agent = self.agents.get("stock_monitoring_agent")
            if stock_agent and hasattr(stock_agent, 'inventory_data'):
                inventory = {}
                for key, inv_data in stock_agent.inventory_data.items():
                    inventory[inv_data.sku] = {
                        'current_stock': inv_data.current_stock,
                        'available_stock': inv_data.available_stock,
                        'reserved_stock': inv_data.reserved_stock,
                        'location': inv_data.location,
                        'last_updated': inv_data.last_updated.isoformat()
                    }
                return jsonify(inventory)
            return jsonify({})
        
        @self.web_app.route('/api/reorder_points')
        def api_reorder_points():
            reorder_agent = self.agents.get("reorder_point_agent")
            if reorder_agent and hasattr(reorder_agent, 'reorder_points'):
                return jsonify(reorder_agent.reorder_points)
            return jsonify({})
        
        @self.web_app.route('/api/safety_stocks')
        def api_safety_stocks():
            safety_agent = self.agents.get("safety_stock_agent")
            if safety_agent and hasattr(safety_agent, 'safety_stocks'):
                return jsonify(safety_agent.safety_stocks)
            return jsonify({})
        
        @self.web_app.route('/api/classifications')
        def api_classifications():
            abc_agent = self.agents.get("abc_classification_agent")
            if abc_agent and hasattr(abc_agent, 'classifications'):
                return jsonify(abc_agent.classifications)
            return jsonify({})
        
        @self.web_app.route('/api/products')
        def api_products():
            products_data = {}
            for sku, product in self.products.items():
                products_data[sku] = {
                    'name': product.name,
                    'category': product.category,
                    'price': product.price,
                    'lead_time': product.lead_time,
                    'supplier_id': product.supplier_id,
                    'abc_class': product.abc_class,
                    'xyz_class': product.xyz_class
                }
            return jsonify(products_data)
        
        @self.web_app.route('/api/control/<action>', methods=['POST'])
        def api_control(action):
            if action == 'start':
                if not self.running:
                    self.start_system()
                    return jsonify({'status': 'success', 'message': 'System started'})
                return jsonify({'status': 'info', 'message': 'System already running'})
            elif action == 'stop':
                if self.running:
                    self.stop_system()
                    return jsonify({'status': 'success', 'message': 'System stopped'})
                return jsonify({'status': 'info', 'message': 'System already stopped'})
            elif action == 'restart':
                self.stop_system()
                time.sleep(2)
                self.start_system()
                return jsonify({'status': 'success', 'message': 'System restarted'})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid action'}), 400
    
    def initialize_system(self):
        """Initialize all agents and the system"""
        self.agents = {
            "demand_forecasting_agent": DemandForecastingAgent("demand_forecasting_agent", self),
            "stock_monitoring_agent": StockLevelMonitoringAgent("stock_monitoring_agent", self),
            "reorder_point_agent": ReorderPointAgent("reorder_point_agent", self),
            "inventory_allocation_agent": InventoryAllocationAgent("inventory_allocation_agent", self),
            "seasonal_adjustment_agent": SeasonalAdjustmentAgent("seasonal_adjustment_agent", self),
            "abc_classification_agent": ABCClassificationAgent("abc_classification_agent", self),
            "safety_stock_agent": SafetyStockOptimizationAgent("safety_stock_agent", self)
        }
        
        self.initialize_sample_data()
        print("Inventory Optimization System initialized with agents:")
        for agent_id in self.agents.keys():
            print(f"  - {agent_id}")
    
    def initialize_sample_data(self):
        """Initialize sample products and historical data"""
        sample_products = [
            Product("SKU001", "Laptop Pro", "Electronics", 1200.00, 5, "SUPPLIER001"),
            Product("SKU002", "Wireless Mouse", "Electronics", 25.00, 3, "SUPPLIER002"),
            Product("SKU003", "Office Chair", "Furniture", 350.00, 10, "SUPPLIER003"),
            Product("SKU004", "Coffee Maker", "Appliances", 150.00, 7, "SUPPLIER001"),
            Product("SKU005", "Desk Lamp", "Furniture", 75.00, 4, "SUPPLIER002")
        ]
        
        for product in sample_products:
            self.products[product.sku] = product
        
        self.generate_sample_historical_data()
    
    def generate_sample_historical_data(self):
        """Generate sample historical data for demonstration"""
        agent = self.agents["demand_forecasting_agent"]
        
        for sku in self.products.keys():
            for day in range(90):
                date = (datetime.now() - timedelta(days=90-day)).strftime('%Y-%m-%d')
                
                base_demand = {'SKU001': 5, 'SKU002': 20, 'SKU003': 3, 'SKU004': 8, 'SKU005': 12}.get(sku, 5)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
                noise = np.random.normal(0, 0.2)
                demand = max(0, int(base_demand * seasonal_factor * (1 + noise)))
                
                agent.update_historical_data({
                    'sku': sku,
                    'date': date,
                    'demand': demand,
                    'price': self.products[sku].price,
                    'promotion': day % 30 == 0,
                    'weather': 'normal'
                })
    
    def start_system(self):
        """Start all agents"""
        print("\nStarting Inventory Optimization System...")
        self.running = True
        
        for agent_id, agent in self.agents.items():
            try:
                agent.start()
                print(f"✓ Started {agent_id}")
            except Exception as e:
                print(f"✗ Failed to start {agent_id}: {e}")
                
        print("All agents startup completed!")
        self.simulate_initial_inventory()
    
    def simulate_initial_inventory(self):
        """Simulate initial inventory levels"""
        stock_agent = self.agents["stock_monitoring_agent"]
        
        for sku in self.products.keys():
            try:
                initial_stock = np.random.randint(50, 200)
                reserved_stock = np.random.randint(0, 20)
                
                stock_agent.update_inventory_data({
                    'sku': sku,
                    'location': 'main',
                    'current_stock': initial_stock,
                    'reserved_stock': reserved_stock
                })
                
            except Exception as e:
                print(f"Error initializing inventory for {sku}: {e}")
    
    def stop_system(self):
        """Stop all agents"""
        print("\nStopping Inventory Optimization System...")
        self.running = False
        
        for agent in self.agents.values():
            agent.stop()
            
        print("All agents stopped successfully!")
    
    def route_message(self, message: Message):
        """Route message between agents"""
        if message.receiver == "all":
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender:
                    agent.receive_message(message)
        elif message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'running': self.running,
            'agents': {}
        }
        
        for agent_id, agent in self.agents.items():
            status['agents'][agent_id] = {
                'running': agent.running,
                'queue_size': agent.message_queue.qsize()
            }
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_products': len(self.products),
            'active_agents': sum(1 for agent in self.agents.values() if agent.running)
        }
        
        demand_agent = self.agents.get("demand_forecasting_agent")
        if demand_agent and hasattr(demand_agent, 'forecast_cache'):
            metrics['forecasts_generated'] = len(demand_agent.forecast_cache)
        
        reorder_agent = self.agents.get("reorder_point_agent")
        if reorder_agent and hasattr(reorder_agent, 'reorder_points'):
            metrics['reorder_points_set'] = len(reorder_agent.reorder_points)
        
        return metrics
    
    def run_web_server(self, port=5000):
        """Run the web server"""
        def open_browser():
            webbrowser.open(f'http://localhost:{port}')
        
        # Open browser after a short delay
        Timer(1.5, open_browser).start()
        
        print(f"\n🌐 Starting web server at http://localhost:{port}")
        print("🔧 Dashboard will open automatically in your browser")
        
        self.web_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ==================== HTML TEMPLATE ====================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Inventory Optimization Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .status-bar {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .status-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            flex: 1;
            min-width: 200px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
        }
        
        .status-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }
        
        .status-value {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }
        
        .status-online {
            color: #10b981;
        }
        
        .status-offline {
            color: #ef4444;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4rem;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }
        
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        
        .agent-card {
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .agent-card.running {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border-color: #10b981;
        }
        
        .agent-card.stopped {
            background: linear-gradient(135deg, #fef2f2, #fecaca);
            border-color: #ef4444;
        }
        
        .agent-name {
            font-weight: 600;
            margin-bottom: 5px;
            font-size: 0.9rem;
        }
        
        .agent-status {
            font-size: 0.8rem;
            padding: 4px 8px;
            border-radius: 15px;
            font-weight: 500;
        }
        
        .status-running {
            background: #10b981;
            color: white;
        }
        
        .status-stopped {
            background: #ef4444;
            color: white;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .data-table th {
            background: #f8fafc;
            font-weight: 600;
            color: #667eea;
        }
        
        .data-table tr:hover {
            background: #f8fafc;
        }
        
        .chart-container {
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8fafc;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .alert-success {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #10b981;
        }
        
        .alert-error {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-top: 5px;
        }
        
        .forecast-chart {
            display: flex;
            align-items: end;
            height: 100px;
            gap: 4px;
            margin-top: 15px;
        }
        
        .forecast-bar {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 2px;
            flex: 1;
            min-height: 10px;
            opacity: 0.8;
            transition: opacity 0.3s ease;
        }
        
        .forecast-bar:hover {
            opacity: 1;
        }
        
        .footer {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin-top: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .last-updated {
            color: #6b7280;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .status-bar {
                flex-direction: column;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Autonomous Inventory Optimization</h1>
            <p>Real-time Multi-Agent System Dashboard</p>
            <div class="controls">
                <button class="btn" onclick="controlSystem('start')">▶️ Start System</button>
                <button class="btn" onclick="controlSystem('stop')">⏹️ Stop System</button>
                <button class="btn" onclick="controlSystem('restart')">🔄 Restart System</button>
                <button class="btn" onclick="refreshData()">🔃 Refresh Data</button>
            </div>
        </div>
        
        <div id="alerts"></div>
        
        <div class="status-bar">
            <div class="status-card">
                <h3>System Status</h3>
                <div class="status-value" id="system-status">
                    <span class="loading"></span>
                </div>
            </div>
            <div class="status-card">
                <h3>Active Agents</h3>
                <div class="status-value" id="active-agents">
                    <span class="loading"></span>
                </div>
            </div>
            <div class="status-card">
                <h3>Total Products</h3>
                <div class="status-value" id="total-products">
                    <span class="loading"></span>
                </div>
            </div>
            <div class="status-card">
                <h3>Forecasts Generated</h3>
                <div class="status-value" id="forecasts-generated">
                    <span class="loading"></span>
                </div>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <div class="card">
                <h2>🎯 Agent Status</h2>
                <div class="agent-grid" id="agent-status">
                    <div class="loading"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Demand Forecasts</h2>
                <div id="forecasts-data">
                    <div class="loading"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>📦 Current Inventory</h2>
                <div id="inventory-data">
                    <div class="loading"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>🎯 Reorder Points</h2>
                <div id="reorder-points-data">
                    <div class="loading"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>🛡️ Safety Stocks</h2>
                <div id="safety-stocks-data">
                    <div class="loading"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>🏷️ Product Classifications</h2>
                <div id="classifications-data">
                    <div class="loading"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="last-updated" id="last-updated">
                Last updated: <span class="loading"></span>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let updateInterval;
        let isAutoRefresh = true;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 Dashboard initialized');
            refreshData();
            startAutoRefresh();
        });
        
        // Auto refresh every 5 seconds
        function startAutoRefresh() {
            if (updateInterval) clearInterval(updateInterval);
            updateInterval = setInterval(() => {
                if (isAutoRefresh) {
                    refreshData();
                }
            }, 5000);
        }
        
        // Refresh all data
        async function refreshData() {
            console.log('🔄 Refreshing dashboard data...');
            
            try {
                await Promise.all([
                    updateSystemStatus(),
                    updateAgentStatus(),
                    updateForecasts(),
                    updateInventory(),
                    updateReorderPoints(),
                    updateSafetyStocks(),
                    updateClassifications()
                ]);
                
                document.getElementById('last-updated').innerHTML = 
                    `Last updated: ${new Date().toLocaleString()}`;
                    
            } catch (error) {
                console.error('❌ Error refreshing data:', error);
                showAlert('Error refreshing data: ' + error.message, 'error');
            }
        }
        
        // Update system status
        async function updateSystemStatus() {
            try {
                const [statusResponse, metricsResponse] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/metrics')
                ]);
                
                const status = await statusResponse.json();
                const metrics = await metricsResponse.json();
                
                // Update status indicators
                document.getElementById('system-status').innerHTML = 
                    `<span class="${status.running ? 'status-online' : 'status-offline'}">
                        ${status.running ? '🟢 Online' : '🔴 Offline'}
                    </span>`;
                
                document.getElementById('active-agents').innerHTML = 
                    `<span class="metric-value">${metrics.active_agents || 0}</span>`;
                
                document.getElementById('total-products').innerHTML = 
                    `<span class="metric-value">${metrics.total_products || 0}</span>`;
                
                document.getElementById('forecasts-generated').innerHTML = 
                    `<span class="metric-value">${metrics.forecasts_generated || 0}</span>`;
                
            } catch (error) {
                console.error('Error updating system status:', error);
            }
        }
        
        // Update agent status
        async function updateAgentStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const agentStatusDiv = document.getElementById('agent-status');
                let html = '';
                
                const agentNames = {
                    'demand_forecasting_agent': 'Demand Forecasting',
                    'stock_monitoring_agent': 'Stock Monitoring',
                    'reorder_point_agent': 'Reorder Point',
                    'inventory_allocation_agent': 'Inventory Allocation',
                    'seasonal_adjustment_agent': 'Seasonal Adjustment',
                    'abc_classification_agent': 'ABC Classification',
                    'safety_stock_agent': 'Safety Stock'
                };
                
                for (const [agentId, agentData] of Object.entries(data.agents || {})) {
                    const isRunning = agentData.running;
                    const queueSize = agentData.queue_size || 0;
                    
                    html += `
                        <div class="agent-card ${isRunning ? 'running' : 'stopped'}">
                            <div class="agent-name">${agentNames[agentId] || agentId}</div>
                            <div class="agent-status ${isRunning ? 'status-running' : 'status-stopped'}">
                                ${isRunning ? '🟢 Running' : '🔴 Stopped'}
                            </div>
                            <div style="font-size: 0.7rem; margin-top: 5px; color: #666;">
                                Queue: ${queueSize}
                            </div>
                        </div>
                    `;
                }
                
                agentStatusDiv.innerHTML = html;
                
            } catch (error) {
                console.error('Error updating agent status:', error);
                document.getElementById('agent-status').innerHTML = 
                    '<div style="color: #ef4444;">Error loading agent status</div>';
            }
        }
        
        // Update forecasts
        async function updateForecasts() {
            try {
                const response = await fetch('/api/forecasts');
                const forecasts = await response.json();
                
                const forecastsDiv = document.getElementById('forecasts-data');
                
                if (Object.keys(forecasts).length === 0) {
                    forecastsDiv.innerHTML = '<div style="color: #666;">No forecasts available yet...</div>';
                    return;
                }
                
                let html = '<table class="data-table"><thead><tr><th>SKU</th><th>Next 7 Days Avg</th><th>Accuracy</th><th>Updated</th></tr></thead><tbody>';
                
                for (const [sku, forecast] of Object.entries(forecasts)) {
                    const avgDemand = forecast.predicted_demand.reduce((a, b) => a + b, 0) / forecast.predicted_demand.length;
                    const accuracy = (forecast.accuracy_score * 100).toFixed(1);
                    const updated = new Date(forecast.timestamp).toLocaleString();
                    
                    html += `
                        <tr>
                            <td><strong>${sku}</strong></td>
                            <td>${avgDemand.toFixed(1)} units</td>
                            <td>${accuracy}%</td>
                            <td>${updated}</td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                forecastsDiv.innerHTML = html;
                
            } catch (error) {
                console.error('Error updating forecasts:', error);
                document.getElementById('forecasts-data').innerHTML = 
                    '<div style="color: #ef4444;">Error loading forecasts</div>';
            }
        }
        
        // Update inventory
        async function updateInventory() {
            try {
                const response = await fetch('/api/inventory');
                const inventory = await response.json();
                
                const inventoryDiv = document.getElementById('inventory-data');
                
                if (Object.keys(inventory).length === 0) {
                    inventoryDiv.innerHTML = '<div style="color: #666;">No inventory data available...</div>';
                    return;
                }
                
                let html = '<table class="data-table"><thead><tr><th>SKU</th><th>Current</th><th>Available</th><th>Reserved</th><th>Updated</th></tr></thead><tbody>';
                
                for (const [sku, inv] of Object.entries(inventory)) {
                    const stockLevel = inv.current_stock;
                    const stockColor = stockLevel < 50 ? '#ef4444' : stockLevel < 100 ? '#f59e0b' : '#10b981';
                    const updated = new Date(inv.last_updated).toLocaleString();
                    
                    html += `
                        <tr>
                            <td><strong>${sku}</strong></td>
                            <td style="color: ${stockColor}; font-weight: bold;">${inv.current_stock}</td>
                            <td>${inv.available_stock}</td>
                            <td>${inv.reserved_stock}</td>
                            <td>${updated}</td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                inventoryDiv.innerHTML = html;
                
            } catch (error) {
                console.error('Error updating inventory:', error);
                document.getElementById('inventory-data').innerHTML = 
                    '<div style="color: #ef4444;">Error loading inventory</div>';
            }
        }
        
        // Update reorder points
        async function updateReorderPoints() {
            try {
                const response = await fetch('/api/reorder_points');
                const reorderPoints = await response.json();
                
                const reorderDiv = document.getElementById('reorder-points-data');
                
                if (Object.keys(reorderPoints).length === 0) {
                    reorderDiv.innerHTML = '<div style="color: #666;">No reorder points calculated yet...</div>';
                    return;
                }
                
                let html = '<table class="data-table"><thead><tr><th>SKU</th><th>Reorder Point</th><th>Status</th></tr></thead><tbody>';
                
                for (const [sku, rop] of Object.entries(reorderPoints)) {
                    html += `
                        <tr>
                            <td><strong>${sku}</strong></td>
                            <td>${rop.toFixed(1)} units</td>
                            <td><span style="color: #10b981;">✓ Calculated</span></td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                reorderDiv.innerHTML = html;
                
            } catch (error) {
                console.error('Error updating reorder points:', error);
                document.getElementById('reorder-points-data').innerHTML = 
                    '<div style="color: #ef4444;">Error loading reorder points</div>';
            }
        }
        
        // Update safety stocks
        async function updateSafetyStocks() {
            try {
                const response = await fetch('/api/safety_stocks');
                const safetyStocks = await response.json();
                
                const safetyDiv = document.getElementById('safety-stocks-data');
                
                if (Object.keys(safetyStocks).length === 0) {
                    safetyDiv.innerHTML = '<div style="color: #666;">No safety stocks calculated yet...</div>';
                    return;
                }
                
                let html = '<table class="data-table"><thead><tr><th>SKU</th><th>Safety Stock</th><th>Status</th></tr></thead><tbody>';
                
                for (const [sku, ss] of Object.entries(safetyStocks)) {
                    html += `
                        <tr>
                            <td><strong>${sku}</strong></td>
                            <td>${ss.toFixed(1)} units</td>
                            <td><span style="color: #10b981;">✓ Optimized</span></td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                safetyDiv.innerHTML = html;
                
            } catch (error) {
                console.error('Error updating safety stocks:', error);
                document.getElementById('safety-stocks-data').innerHTML = 
                    '<div style="color: #ef4444;">Error loading safety stocks</div>';
            }
        }
        
        // Update classifications
        async function updateClassifications() {
            try {
                const response = await fetch('/api/classifications');
                const classifications = await response.json();
                
                const classDiv = document.getElementById('classifications-data');
                
                if (Object.keys(classifications).length === 0) {
                    classDiv.innerHTML = '<div style="color: #666;">No classifications available yet...</div>';
                    return;
                }
                
                let html = '<table class="data-table"><thead><tr><th>SKU</th><th>ABC Class</th><th>XYZ Class</th><th>Priority</th></tr></thead><tbody>';
                
                for (const [sku, cls] of Object.entries(classifications)) {
                    const abcColor = cls.abc_class === 'A' ? '#10b981' : cls.abc_class === 'B' ? '#f59e0b' : '#6b7280';
                    const xyzColor = cls.xyz_class === 'X' ? '#10b981' : cls.xyz_class === 'Y' ? '#f59e0b' : '#6b7280';
                    const priority = cls.abc_class === 'A' ? 'High' : cls.abc_class === 'B' ? 'Medium' : 'Low';
                    
                    html += `
                        <tr>
                            <td><strong>${sku}</strong></td>
                            <td><span style="color: ${abcColor}; font-weight: bold;">${cls.abc_class}</span></td>
                            <td><span style="color: ${xyzColor}; font-weight: bold;">${cls.xyz_class}</span></td>
                            <td>${priority}</td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                classDiv.innerHTML = html;
                
            } catch (error) {
                console.error('Error updating classifications:', error);
                document.getElementById('classifications-data').innerHTML = 
                    '<div style="color: #ef4444;">Error loading classifications</div>';
            }
        }
        
        // Control system
        async function controlSystem(action) {
            try {
                showAlert(`${action.charAt(0).toUpperCase() + action.slice(1)}ing system...`, 'success');
                
                const response = await fetch(`/api/control/${action}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    showAlert(result.message, 'success');
                    setTimeout(refreshData, 1000);
                } else {
                    showAlert(result.message, result.status === 'error' ? 'error' : 'success');
                }
                
            } catch (error) {
                console.error('Error controlling system:', error);
                showAlert('Error controlling system: ' + error.message, 'error');
            }
        }
        
        // Show alert
        function showAlert(message, type) {
            const alertsDiv = document.getElementById('alerts');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;
            
            alertsDiv.appendChild(alertDiv);
            
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
        
        // Utility functions
        function formatNumber(num) {
            return new Intl.NumberFormat().format(num);
        }
        
        function formatCurrency(num) {
            return new Intl.NumberFormat('en-US', { 
                style: 'currency', 
                currency: 'USD' 
            }).format(num);
        }
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        });
        
        console.log('📊 Dashboard script loaded successfully');
    </script>
</body>
</html>
"""

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run the system with web interface"""
    system = InventoryOptimizationSystem()
    
    try:
        print("="*60)
        print("🤖 AUTONOMOUS INVENTORY OPTIMIZATION SYSTEM")
        print("="*60)
        
        # Initialize the system
        system.initialize_system()
        
        # Start the agents
        system.start_system()
        
        # Wait a moment for agents to initialize
        time.sleep(2)
        
        print("\n✅ System initialized and running!")
        print("🌐 Starting web dashboard...")
        
        # Run the web server (this will block)
        system.run_web_server(port=5000)
        
    except KeyboardInterrupt:
        print("\n⚠️ Received interrupt signal...")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        # Stop the system
        print("\n🔄 Shutting down system...")
        system.stop_system()
        print("✅ System shutdown complete.")

if __name__ == "__main__":
    main()