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

# External libraries for ML (would be imported in real implementation)
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
                    message = self.message_queue.get_nowait()
                    self.handle_message(message)
                
                # Perform main agent tasks
                self.execute_main_task()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in agent {self.agent_id}: {e}")
    
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
           (current_time - self.last_forecast_time).seconds > 3600:
            
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
        
        for i in range(7, len(data)):  # Use 7-day lookback
            # Temporal features
            current_date = data.iloc[i]['date']
            day_of_week = current_date.dayofweek
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
            day_of_week = future_date.dayofweek
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
            prediction = max(0, prediction)  # Ensure non-negative
            
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
            location="main",  # Simplified for this example
            forecast_horizon=horizon,
            predicted_demand=forecasts,
            confidence_interval=(confidence_lower, confidence_upper),
            accuracy_score=1 / (1 + mae),  # Simplified accuracy score
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

# ==================== STOCK LEVEL MONITORING AGENT ====================

class StockLevelMonitoringAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.inventory_data = {}
        self.alert_thresholds = {}
        self.anomaly_detector = None
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.INVENTORY_STATUS.value:
            self.update_inventory_data(message.content)
        elif message.message_type == MessageType.REORDER_ALERT.value:
            self.process_reorder_alert(message.content)
    
    def execute_main_task(self):
        """Monitor stock levels and detect anomalies"""
        self.check_stock_levels()
        self.detect_anomalies()
        self.broadcast_status_updates()
    
    def update_inventory_data(self, data: Dict[str, Any]):
        """Update inventory data from external systems"""
        key = f"{data['sku']}_{data['location']}"
        self.inventory_data[key] = InventoryData(
            sku=data['sku'],
            location=data['location'],
            current_stock=data['current_stock'],
            reserved_stock=data.get('reserved_stock', 0),
            available_stock=data['current_stock'] - data.get('reserved_stock', 0),
            last_updated=datetime.now()
        )
    
    def check_stock_levels(self):
        """Check current stock levels against thresholds"""
        for key, inventory in self.inventory_data.items():
            sku = inventory.sku
            
            # Get reorder point from reorder point agent
            if f"{sku}_reorder_point" in self.alert_thresholds:
                reorder_point = self.alert_thresholds[f"{sku}_reorder_point"]
                
                if inventory.available_stock <= reorder_point:
                    self.send_message(
                        receiver="reorder_point_agent",
                        message_type=MessageType.REORDER_ALERT.value,
                        content={
                            "sku": sku,
                            "location": inventory.location,
                            "current_stock": inventory.current_stock,
                            "available_stock": inventory.available_stock,
                            "reorder_point": reorder_point,
                            "urgency": "high" if inventory.available_stock <= reorder_point * 0.5 else "medium"
                        },
                        priority=2
                    )
    
    def detect_anomalies(self):
        """Detect anomalies in stock movement patterns"""
        for key, inventory in self.inventory_data.items():
            # Simple anomaly detection based on stock change patterns
            if hasattr(self, 'previous_inventory_data') and \
               key in self.previous_inventory_data:
                
                prev_stock = self.previous_inventory_data[key].current_stock
                current_stock = inventory.current_stock
                stock_change = prev_stock - current_stock
                
                # Check for unusual stock changes
                if abs(stock_change) > 100:  # Threshold for anomaly
                    self.send_message(
                        receiver="all",
                        message_type=MessageType.INVENTORY_STATUS.value,
                        content={
                            "type": "anomaly_detected",
                            "sku": inventory.sku,
                            "location": inventory.location,
                            "stock_change": stock_change,
                            "timestamp": datetime.now().isoformat()
                        },
                        priority=3
                    )
        
        # Store current data for next comparison
        self.previous_inventory_data = self.inventory_data.copy()
    
    def broadcast_status_updates(self):
        """Broadcast inventory status to other agents"""
        for key, inventory in self.inventory_data.items():
            self.send_message(
                receiver="all",
                message_type=MessageType.INVENTORY_STATUS.value,
                content={
                    "type": "status_update",
                    "sku": inventory.sku,
                    "location": inventory.location,
                    "current_stock": inventory.current_stock,
                    "available_stock": inventory.available_stock,
                    "last_updated": inventory.last_updated.isoformat()
                }
            )
    
    def process_reorder_alert(self, alert_data: Dict[str, Any]):
        """Process reorder alerts from other agents"""
        # Log the alert and update monitoring parameters
        sku = alert_data['sku']
        self.alert_thresholds[f"{sku}_last_alert"] = datetime.now()

# ==================== REORDER POINT AGENT ====================

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
        elif message.message_type == MessageType.REORDER_ALERT.value:
            self.process_reorder_request(message.content)
    
    def execute_main_task(self):
        """Calculate and update reorder points"""
        self.calculate_reorder_points()
        self.generate_reorder_recommendations()
    
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
        # Get average daily demand from forecast
        if sku not in self.demand_forecasts:
            return 0
        
        forecast = self.demand_forecasts[sku]
        predicted_demand = forecast.get('predicted_demand', [])
        
        if not predicted_demand:
            return 0
        
        avg_daily_demand = np.mean(predicted_demand[:7])  # Use first week
        
        # Get lead time (default to 7 days if not available)
        lead_time = self.lead_times.get(sku, 7)
        
        # Get safety stock
        safety_stock = self.safety_stocks.get(sku, avg_daily_demand * 2)
        
        # Calculate reorder point: (Average Daily Demand × Lead Time) + Safety Stock
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        
        return max(0, reorder_point)
    
    def calculate_reorder_points(self):
        """Calculate reorder points for all SKUs"""
        for sku in self.demand_forecasts.keys():
            reorder_point = self.calculate_reorder_point(sku)
            self.reorder_points[sku] = reorder_point
            
            # Notify stock monitoring agent of updated reorder point
            self.send_message(
                receiver="stock_monitoring_agent",
                message_type=MessageType.INVENTORY_STATUS.value,
                content={
                    "type": "reorder_point_update",
                    "sku": sku,
                    "reorder_point": reorder_point
                }
            )
    
    def generate_reorder_recommendations(self):
        """Generate reorder recommendations"""
        current_time = datetime.now()
        
        for sku, reorder_point in self.reorder_points.items():
            # This would typically get current stock from inventory system
            # For simulation, we'll assume we have this data
            current_stock = 50  # Placeholder
            
            if current_stock <= reorder_point:
                # Calculate recommended order quantity (simplified EOQ)
                avg_daily_demand = np.mean(self.demand_forecasts[sku]['predicted_demand'][:7])
                recommended_qty = max(int(avg_daily_demand * 14), 1)  # 2 weeks supply
                
                recommendation = ReorderRecommendation(
                    sku=sku,
                    location="main",
                    current_stock=current_stock,
                    reorder_point=reorder_point,
                    recommended_quantity=recommended_qty,
                    urgency_level="high" if current_stock <= reorder_point * 0.5 else "medium",
                    reasoning=f"Stock level ({current_stock}) below reorder point ({reorder_point:.1f})",
                    timestamp=current_time
                )
                
                # Send recommendation to inventory allocation agent
                self.send_message(
                    receiver="inventory_allocation_agent",
                    message_type=MessageType.ALLOCATION_REQUEST.value,
                    content={
                        "sku": sku,
                        "recommended_quantity": recommended_qty,
                        "urgency": recommendation.urgency_level,
                        "current_stock": current_stock,
                        "reorder_point": reorder_point
                    }
                )
    
    def process_reorder_request(self, request_data: Dict[str, Any]):
        """Process reorder requests from other agents"""
        sku = request_data['sku']
        current_stock = request_data['current_stock']
        
        if sku in self.reorder_points:
            reorder_point = self.reorder_points[sku]
            
            if current_stock <= reorder_point:
                # Confirm reorder is needed and calculate quantity
                avg_daily_demand = np.mean(self.demand_forecasts[sku]['predicted_demand'][:7])
                recommended_qty = max(int(avg_daily_demand * 14), 1)
                
                self.send_message(
                    receiver="inventory_allocation_agent",
                    message_type=MessageType.ALLOCATION_REQUEST.value,
                    content={
                        "sku": sku,
                        "recommended_quantity": recommended_qty,
                        "urgency": request_data.get('urgency', 'medium'),
                        "confirmation": True
                    }
                )

# ==================== INVENTORY ALLOCATION AGENT ====================

class InventoryAllocationAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.allocation_rules = {}
        self.supplier_data = {}
        self.pending_orders = {}
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.ALLOCATION_REQUEST.value:
            self.process_allocation_request(message.content)
        elif message.message_type == MessageType.INVENTORY_STATUS.value:
            self.update_inventory_status(message.content)
    
    def execute_main_task(self):
        """Execute allocation optimization"""
        self.optimize_allocations()
        self.process_pending_orders()
    
    def process_allocation_request(self, request_data: Dict[str, Any]):
        """Process allocation requests from reorder point agent"""
        sku = request_data['sku']
        quantity = request_data['recommended_quantity']
        urgency = request_data['urgency']
        
        # Multi-objective optimization for allocation
        allocation_plan = self.optimize_allocation(sku, quantity, urgency)
        
        if allocation_plan:
            # Execute allocation
            self.execute_allocation(allocation_plan)
            
            # Log the allocation
            self.pending_orders[f"{sku}_{datetime.now().isoformat()}"] = {
                'sku': sku,
                'quantity': quantity,
                'allocation_plan': allocation_plan,
                'status': 'pending',
                'timestamp': datetime.now()
            }
    
    def optimize_allocation(self, sku: str, quantity: int, urgency: str) -> Dict[str, Any]:
        """Optimize allocation considering multiple objectives"""
        # Simplified multi-objective optimization
        
        # Get available suppliers
        available_suppliers = self.get_available_suppliers(sku)
        
        if not available_suppliers:
            return None
        
        # Score suppliers based on multiple criteria
        supplier_scores = {}
        for supplier in available_suppliers:
            cost_score = 1 / (supplier.get('cost_per_unit', 1) + 1)  # Lower cost = higher score
            reliability_score = supplier.get('reliability', 0.8)
            lead_time_score = 1 / (supplier.get('lead_time', 7) + 1)  # Shorter lead time = higher score
            
            # Weight scores based on urgency
            if urgency == 'high':
                total_score = 0.2 * cost_score + 0.3 * reliability_score + 0.5 * lead_time_score
            else:
                total_score = 0.4 * cost_score + 0.4 * reliability_score + 0.2 * lead_time_score
            
            supplier_scores[supplier['id']] = total_score
        
        # Select best supplier
        best_supplier_id = max(supplier_scores, key=supplier_scores.get)
        best_supplier = next(s for s in available_suppliers if s['id'] == best_supplier_id)
        
        allocation_plan = {
            'sku': sku,
            'quantity': quantity,
            'supplier_id': best_supplier_id,
            'cost_per_unit': best_supplier['cost_per_unit'],
            'total_cost': quantity * best_supplier['cost_per_unit'],
            'expected_delivery': datetime.now() + timedelta(days=best_supplier['lead_time']),
            'urgency': urgency
        }
        
        return allocation_plan
    
    def get_available_suppliers(self, sku: str) -> List[Dict[str, Any]]:
        """Get available suppliers for a SKU"""
        # Mock supplier data
        return [
            {
                'id': 'supplier_1',
                'name': 'Fast Supplier',
                'cost_per_unit': 10.50,
                'lead_time': 3,
                'reliability': 0.95,
                'capacity': 1000
            },
            {
                'id': 'supplier_2',
                'name': 'Cheap Supplier',
                'cost_per_unit': 8.75,
                'lead_time': 7,
                'reliability': 0.85,
                'capacity': 2000
            },
            {
                'id': 'supplier_3',
                'name': 'Reliable Supplier',
                'cost_per_unit': 12.00,
                'lead_time': 5,
                'reliability': 0.98,
                'capacity': 1500
            }
        ]
    
    def execute_allocation(self, allocation_plan: Dict[str, Any]):
        """Execute the allocation plan"""
        # In a real system, this would place orders with suppliers
        print(f"Executing allocation: {allocation_plan['quantity']} units of {allocation_plan['sku']} "
              f"from {allocation_plan['supplier_id']} at ${allocation_plan['cost_per_unit']:.2f}/unit")
        
        # Notify other agents
        self.send_message(
            receiver="all",
            message_type=MessageType.ALLOCATION_REQUEST.value,
            content={
                "type": "allocation_executed",
                "allocation_plan": allocation_plan,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def optimize_allocations(self):
        """Optimize all pending allocations"""
        # This could implement more sophisticated allocation optimization
        pass
    
    def process_pending_orders(self):
        """Process and update pending orders"""
        current_time = datetime.now()
        
        for order_id, order in list(self.pending_orders.items()):
            # Check if order should have been delivered
            expected_delivery = order['allocation_plan']['expected_delivery']
            
            if current_time >= expected_delivery and order['status'] == 'pending':
                # Mark as delivered (in real system, this would be based on actual delivery confirmation)
                order['status'] = 'delivered'
                
                # Update inventory levels (notify stock monitoring agent)
                self.send_message(
                    receiver="stock_monitoring_agent",
                    message_type=MessageType.INVENTORY_STATUS.value,
                    content={
                        "type": "inventory_update",
                        "sku": order['sku'],
                        "quantity_added": order['quantity'],
                        "timestamp": current_time.isoformat()
                    }
                )
    
    def update_inventory_status(self, status_data: Dict[str, Any]):
        """Update inventory status from monitoring agent"""
        # Process inventory status updates
        pass

# ==================== SEASONAL ADJUSTMENT AGENT ====================

class SeasonalAdjustmentAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.seasonal_patterns = {}
        self.adjustment_factors = {}
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.DEMAND_UPDATE.value:
            self.analyze_seasonal_patterns(message.content)
    
    def execute_main_task(self):
        """Analyze seasonal patterns and generate adjustments"""
        self.update_seasonal_factors()
        self.broadcast_seasonal_adjustments()
    
    def analyze_seasonal_patterns(self, demand_data: Dict[str, Any]):
        """Analyze seasonal patterns in demand data"""
        sku = demand_data['sku']
        
        if sku not in self.seasonal_patterns:
            self.seasonal_patterns[sku] = {
                'monthly_factors': {},
                'weekly_factors': {},
                'daily_factors': {}
            }
        
        # Calculate seasonal factors based on current month/week/day
        current_date = datetime.now()
        month = current_date.month
        week_of_year = current_date.isocalendar()[1]
        day_of_week = current_date.weekday()
        
        # Update monthly factors
        if month not in self.seasonal_patterns[sku]['monthly_factors']:
            self.seasonal_patterns[sku]['monthly_factors'][month] = []
        
        # In a real system, this would analyze historical patterns
        # For simulation, we'll use predefined seasonal factors
        seasonal_factors = {
            # Holiday season boost
            11: 1.3, 12: 1.5, 1: 0.8,  # Nov, Dec, Jan
            # Summer factors
            6: 1.1, 7: 1.2, 8: 1.1,   # Jun, Jul, Aug
            # Spring factors
            3: 1.0, 4: 1.1, 5: 1.0,   # Mar, Apr, May
            # Other months
            2: 0.9, 9: 0.95, 10: 1.05
        }
        
        self.adjustment_factors[sku] = seasonal_factors.get(month, 1.0)
    
    def update_seasonal_factors(self):
        """Update seasonal adjustment factors"""
        current_month = datetime.now().month
        
        # Define seasonal patterns (this would typically be learned from historical data)
        base_seasonal_factors = {
            1: 0.8,   # January - post-holiday drop
            2: 0.85,  # February - low season
            3: 0.95,  # March - spring pickup
            4: 1.0,   # April - normal
            5: 1.05,  # May - spring peak
            6: 1.1,   # June - summer start
            7: 1.15,  # July - summer peak
            8: 1.1,   # August - summer end
            9: 0.95,  # September - back to school
            10: 1.0,  # October - normal
            11: 1.2,  # November - holiday start
            12: 1.4   # December - holiday peak
        }
        
        # Update factors for all SKUs
        for sku in self.seasonal_patterns.keys():
            base_factor = base_seasonal_factors.get(current_month, 1.0)
            
            # Add some product-specific variation
            if 'electronics' in sku.lower():
                base_factor *= 1.1  # Electronics see higher holiday boost
            elif 'clothing' in sku.lower():
                base_factor *= 0.95 if current_month in [1, 2] else 1.0  # Clothing slower in winter
            
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

# ==================== ABC CLASSIFICATION AGENT ====================

class ABCClassificationAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.product_data = {}
        self.classifications = {}
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.INVENTORY_STATUS.value:
            self.update_product_data(message.content)
        elif message.message_type == MessageType.DEMAND_UPDATE.value:
            self.update_demand_data(message.content)
    
    def execute_main_task(self):
        """Perform ABC classification analysis"""
        self.perform_abc_analysis()
        self.perform_xyz_analysis()
        self.broadcast_classifications()
    
    def update_product_data(self, data: Dict[str, Any]):
        """Update product performance data"""
        if data.get('type') == 'status_update':
            sku = data['sku']
            if sku not in self.product_data:
                self.product_data[sku] = {
                    'revenue': 0,
                    'volume': 0,
                    'frequency': 0,
                    'demand_variability': 0
                }
            
            # Update with new inventory data
            # In real system, this would be calculated from sales data
            self.product_data[sku]['volume'] += 1  # Simplified tracking
    
    def update_demand_data(self, data: Dict[str, Any]):
        """Update demand data for analysis"""
        sku = data['sku']
        if sku not in self.product_data:
            self.product_data[sku] = {
                'revenue': 0,
                'volume': 0,
                'frequency': 0,
                'demand_variability': 0
            }
        
        # Calculate demand variability from forecast
        if 'forecast' in data and 'predicted_demand' in data['forecast']:
            predicted_demand = data['forecast']['predicted_demand']
            if len(predicted_demand) > 1:
                self.product_data[sku]['demand_variability'] = np.std(predicted_demand)
    
    def perform_abc_analysis(self):
        """Perform ABC analysis based on revenue/volume"""
        if not self.product_data:
            return
        
        # Calculate revenue for each product (simplified)
        product_revenues = {}
        for sku, data in self.product_data.items():
            # In real system, this would be actual revenue data
            revenue = data['volume'] * 100  # Simplified revenue calculation
            product_revenues[sku] = revenue
        
        # Sort products by revenue (descending)
        sorted_products = sorted(product_revenues.items(), key=lambda x: x[1], reverse=True)
        total_revenue = sum(product_revenues.values())
        
        if total_revenue == 0:
            return
        
        # Classify products
        cumulative_revenue = 0
        for i, (sku, revenue) in enumerate(sorted_products):
            cumulative_revenue += revenue
            cumulative_percentage = cumulative_revenue / total_revenue
            
            if cumulative_percentage <= 0.8:  # Top 80% of revenue
                abc_class = 'A'
            elif cumulative_percentage <= 0.95:  # Next 15% of revenue
                abc_class = 'B'
            else:  # Remaining 5% of revenue
                abc_class = 'C'
            
            if sku not in self.classifications:
                self.classifications[sku] = {}
            self.classifications[sku]['abc_class'] = abc_class
    
    def perform_xyz_analysis(self):
        """Perform XYZ analysis based on demand variability"""
        if not self.product_data:
            return
        
        # Calculate coefficient of variation for each product
        variability_data = {}
        for sku, data in self.product_data.items():
            variability = data.get('demand_variability', 0)
            mean_demand = data.get('volume', 1)  # Avoid division by zero
            
            if mean_demand > 0:
                cv = variability / mean_demand  # Coefficient of variation
                variability_data[sku] = cv
        
        # Classify based on variability
        for sku, cv in variability_data.items():
            if cv <= 0.1:  # Low variability
                xyz_class = 'X'
            elif cv <= 0.25:  # Medium variability
                xyz_class = 'Y'
            else:  # High variability
                xyz_class = 'Z'
            
            if sku not in self.classifications:
                self.classifications[sku] = {}
            self.classifications[sku]['xyz_class'] = xyz_class
    
    def broadcast_classifications(self):
        """Broadcast classification updates to other agents"""
        for sku, classification in self.classifications.items():
            self.send_message(
                receiver="all",
                message_type=MessageType.ABC_UPDATE.value,
                content={
                    "sku": sku,
                    "abc_class": classification.get('abc_class', 'C'),
                    "xyz_class": classification.get('xyz_class', 'Z'),
                    "timestamp": datetime.now().isoformat()
                }
            )

# ==================== SAFETY STOCK OPTIMIZATION AGENT ====================

class SafetyStockOptimizationAgent(BaseAgent):
    def __init__(self, agent_id: str, system_manager):
        super().__init__(agent_id, system_manager)
        self.safety_stocks = {}
        self.service_levels = {}
        self.demand_data = {}
        self.lead_time_data = {}
        
    def handle_message(self, message: Message):
        if message.message_type == MessageType.DEMAND_UPDATE.value:
            self.update_demand_data(message.content)
        elif message.message_type == MessageType.ABC_UPDATE.value:
            self.update_classification_data(message.content)
    
    def execute_main_task(self):
        """Optimize safety stock levels"""
        self.calculate_optimal_safety_stocks()
        self.broadcast_safety_stock_updates()
    
    def update_demand_data(self, data: Dict[str, Any]):
        """Update demand data for safety stock calculations"""
        sku = data['sku']
        if 'forecast' in data:
            forecast = data['forecast']
            if 'predicted_demand' in forecast:
                self.demand_data[sku] = {
                    'predicted_demand': forecast['predicted_demand'],
                    'confidence_interval': forecast.get('confidence_interval', (0, 0)),
                    'timestamp': forecast.get('timestamp', datetime.now().isoformat())
                }
    
    def update_classification_data(self, data: Dict[str, Any]):
        """Update product classification data"""
        sku = data['sku']
        abc_class = data['abc_class']
        
        # Set service levels based on ABC classification
        service_level_map = {
            'A': 0.98,  # Class A: 98% service level
            'B': 0.95,  # Class B: 95% service level
            'C': 0.90   # Class C: 90% service level
        }
        
        self.service_levels[sku] = service_level_map.get(abc_class, 0.90)
    
    def calculate_safety_stock(self, sku: str) -> float:
        """Calculate optimal safety stock for a SKU"""
        if sku not in self.demand_data:
            return 0
        
        # Get demand data
        demand_forecast = self.demand_data[sku]['predicted_demand']
        if not demand_forecast:
            return 0
        
        # Calculate demand statistics
        mean_demand = np.mean(demand_forecast)
        std_demand = np.std(demand_forecast)
        
        # Get service level (default to 95%)
        service_level = self.service_levels.get(sku, 0.95)
        
        # Calculate z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Get lead time (default to 7 days)
        lead_time = self.lead_time_data.get(sku, 7)
        
        # Calculate safety stock using formula:
        # SS = z * sqrt(LT * σ²_demand + mean_demand² * σ²_LT)
        # Simplified version (assuming lead time variance is 0):
        safety_stock = z_score * std_demand * np.sqrt(lead_time)
        
        return max(0, safety_stock)
    
    def calculate_optimal_safety_stocks(self):
        """Calculate optimal safety stock levels for all products"""
        for sku in self.demand_data.keys():
            safety_stock = self.calculate_safety_stock(sku)
            
            # Apply multi-objective optimization
            optimized_safety_stock = self.optimize_safety_stock(sku, safety_stock)
            
            self.safety_stocks[sku] = optimized_safety_stock
    
    def optimize_safety_stock(self, sku: str, initial_safety_stock: float) -> float:
        """Optimize safety stock using multi-objective optimization"""
        # Define objective function
        def objective(ss):
            # Holding cost (increases with safety stock)
            holding_cost = ss * 0.25  # 25% annual holding cost
            
            # Stockout cost (decreases with safety stock)
            # Simplified stockout probability calculation
            if sku in self.demand_data:
                demand_std = np.std(self.demand_data[sku]['predicted_demand'])
                stockout_prob = max(0, 1 - (ss / (demand_std + 1)))  # Simplified
                stockout_cost = stockout_prob * 100  # $100 per stockout
            else:
                stockout_cost = 50
            
            return holding_cost + stockout_cost
        
        # Optimize using simple grid search (in real system, use scipy.optimize)
        best_ss = initial_safety_stock
        best_cost = objective(initial_safety_stock)
        
        # Test different safety stock levels
        for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            test_ss = initial_safety_stock * multiplier
            test_cost = objective(test_ss)
            
            if test_cost < best_cost:
                best_cost = test_cost
                best_ss = test_ss
        
        return best_ss
    
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

# ==================== SYSTEM MANAGER ====================

class InventoryOptimizationSystem:
    def __init__(self):
        self.agents = {}
        self.message_router = {}
        self.running = False
        self.products = {}
        self.performance_metrics = {}
        
    def initialize_system(self):
        """Initialize all agents and the system"""
        # Create agents
        self.agents = {
            "demand_forecasting_agent": DemandForecastingAgent("demand_forecasting_agent", self),
            "stock_monitoring_agent": StockLevelMonitoringAgent("stock_monitoring_agent", self),
            "reorder_point_agent": ReorderPointAgent("reorder_point_agent", self),
            "inventory_allocation_agent": InventoryAllocationAgent("inventory_allocation_agent", self),
            "seasonal_adjustment_agent": SeasonalAdjustmentAgent("seasonal_adjustment_agent", self),
            "abc_classification_agent": ABCClassificationAgent("abc_classification_agent", self),
            "safety_stock_agent": SafetyStockOptimizationAgent("safety_stock_agent", self)
        }
        
        # Initialize sample products
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
        
        # Generate sample historical data
        self.generate_sample_historical_data()
    
    def generate_sample_historical_data(self):
        """Generate sample historical data for demonstration"""
        agent = self.agents["demand_forecasting_agent"]
        
        for sku in self.products.keys():
            # Generate 90 days of historical data
            for day in range(90):
                date = (datetime.now() - timedelta(days=90-day)).strftime('%Y-%m-%d')
                
                # Generate demand with some seasonality and randomness
                base_demand = {'SKU001': 5, 'SKU002': 20, 'SKU003': 3, 'SKU004': 8, 'SKU005': 12}.get(sku, 5)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 365)  # Yearly seasonality
                noise = np.random.normal(0, 0.2)  # Random noise
                demand = max(0, int(base_demand * seasonal_factor * (1 + noise)))
                
                agent.update_historical_data({
                    'sku': sku,
                    'date': date,
                    'demand': demand,
                    'price': self.products[sku].price,
                    'promotion': day % 30 == 0,  # Promotion every 30 days
                    'weather': 'normal'
                })
    
    def start_system(self):
        """Start all agents"""
        print("\nStarting Inventory Optimization System...")
        self.running = True
        
        for agent in self.agents.values():
            agent.start()
            
        print("All agents started successfully!")
        
        # Initialize inventory data
        self.simulate_initial_inventory()
    
    def simulate_initial_inventory(self):
        """Simulate initial inventory levels"""
        stock_agent = self.agents["stock_monitoring_agent"]
        
        for sku in self.products.keys():
            # Simulate initial stock levels
            initial_stock = np.random.randint(50, 200)
            
            stock_agent.update_inventory_data({
                'sku': sku,
                'location': 'main',
                'current_stock': initial_stock,
                'reserved_stock': np.random.randint(0, 20)
            })
    
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
            # Broadcast to all agents except sender
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender:
                    agent.receive_message(message)
        elif message.receiver in self.agents:
            # Send to specific agent
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
        
        # Get agent-specific metrics
        demand_agent = self.agents["demand_forecasting_agent"]
        if hasattr(demand_agent, 'forecast_cache'):
            metrics['forecasts_generated'] = len(demand_agent.forecast_cache)
        
        reorder_agent = self.agents["reorder_point_agent"]
        if hasattr(reorder_agent, 'reorder_points'):
            metrics['reorder_points_set'] = len(reorder_agent.reorder_points)
        
        allocation_agent = self.agents["inventory_allocation_agent"]
        if hasattr(allocation_agent, 'pending_orders'):
            metrics['pending_orders'] = len(allocation_agent.pending_orders)
        
        return metrics
    
    def run_simulation(self, duration_minutes: int = 1):
        """Run system simulation for specified duration"""
        print(f"\nRunning simulation for {duration_minutes} minute(s)...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time and self.running:
            # Print periodic status updates
            print(f"System Status: {self.get_system_status()}")
            print(f"Performance Metrics: {self.get_performance_metrics()}")
            print("-" * 50)
            
            time.sleep(10)  # Update every 10 seconds
        
        print("Simulation completed!")

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to demonstrate the system"""
    # Create and initialize the system
    system = InventoryOptimizationSystem()
    system.initialize_system()
    
    try:
        # Start the system
        system.start_system()
        
        # Run simulation
        system.run_simulation(duration_minutes=1)
        
        # Display final results
        print("\n" + "="*60)
        print("FINAL SYSTEM STATE")
        print("="*60)
        
        print("\nFinal Status:")
        print(json.dumps(system.get_system_status(), indent=2))
        
        print("\nFinal Performance Metrics:")
        print(json.dumps(system.get_performance_metrics(), indent=2))
        
        # Display some agent-specific results
        demand_agent = system.agents["demand_forecasting_agent"]
        if hasattr(demand_agent, 'forecast_cache') and demand_agent.forecast_cache:
            print(f"\nGenerated forecasts for {len(demand_agent.forecast_cache)} products:")
            for sku, forecast in demand_agent.forecast_cache.items():
                print(f"  {sku}: Next 7 days avg demand = {np.mean(forecast.predicted_demand[:7]):.1f}")
        
        reorder_agent = system.agents["reorder_point_agent"]
        if hasattr(reorder_agent, 'reorder_points') and reorder_agent.reorder_points:
            print(f"\nReorder points calculated:")
            for sku, rop in reorder_agent.reorder_points.items():
                print(f"  {sku}: ROP = {rop:.1f}")
        
        safety_stock_agent = system.agents["safety_stock_agent"]
        if hasattr(safety_stock_agent, 'safety_stocks') and safety_stock_agent.safety_stocks:
            print(f"\nSafety stocks optimized:")
            for sku, ss in safety_stock_agent.safety_stocks.items():
                print(f"  {sku}: Safety Stock = {ss:.1f}")
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    finally:
        # Stop the system
        system.stop_system()
        print("\nSystem shutdown complete.")

if __name__ == "__main__":
    main()