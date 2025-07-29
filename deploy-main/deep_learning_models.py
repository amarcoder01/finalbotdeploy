"""
Deep Learning Models for Advanced Trading
LSTM, Transformer, BERT, and Reinforcement Learning implementations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import sklearn
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    DL_AVAILABLE = True
    
    # Try to import TensorFlow, but don't fail if it's not available
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        # Create dummy TensorFlow classes
        class keras:
            class Sequential:
                pass
            class optimizers:
                class Adam:
                    pass
        class layers:
            class Dense:
                pass
            class Dropout:
                pass
except ImportError:
    DL_AVAILABLE = False
    TF_AVAILABLE = False
    # Create dummy classes for when DL libraries are not available
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class LSTM:
            pass
        class Dropout:
            pass
        class TransformerEncoderLayer:
            pass
        class TransformerEncoder:
            pass
    class optim:
        class Adam:
            pass
    class torch:
        def zeros(*args, **kwargs):
            return None
        def FloatTensor(*args, **kwargs):
            return None
        def no_grad():
            return None
        def tanh(*args, **kwargs):
            return None
        def arange(*args, **kwargs):
            return None
        def exp(*args, **kwargs):
            return None
        def sin(*args, **kwargs):
            return None
        def cos(*args, **kwargs):
            return None
    class keras:
        class Sequential:
            pass
        class optimizers:
            class Adam:
                pass
    class layers:
        class Dense:
            pass
        class Dropout:
            pass
    class MinMaxScaler:
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
        def inverse_transform(self, X):
            return X

from logger import logger
from openai_service import OpenAIService

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._create_positional_encoding(d_model, 1000)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, d_model: int, max_len: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take the last output
        x = self.output_projection(x)
        return x




class ReinforcementLearningAgent:
    """Reinforcement Learning agent for trading strategies"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.max_memory = 10000
        
        if TF_AVAILABLE:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
    
    def _build_model(self):
        """Build DQN model"""
        if not TF_AVAILABLE:
            return None
            
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def update_target_model(self):
        """Update target model weights"""
        if hasattr(self, 'target_model') and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if not TF_AVAILABLE or not hasattr(self, 'model'):
            return np.random.randint(0, self.action_size)
        
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int = 32):
        """Train the model on past experiences"""
        if not TF_AVAILABLE or not hasattr(self, 'model') or len(self.memory) < batch_size:
            return
        
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        next_targets = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + 0.95 * np.amax(next_targets[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DeepLearningService:
    """Main service for deep learning models"""
    
    def __init__(self):
        self.lstm_model = None
        self.transformer_model = None
        self.rl_agent = None
        self.scaler = MinMaxScaler()
        self.models_trained = False
        
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for deep learning models"""
        try:
            # Create features
            features = df[['Close', 'Volume', 'High', 'Low', 'Open']].values
            features_scaled = self.scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y.append(features_scaled[i, 0])  # Predict close price
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def train_lstm_model(self, df: pd.DataFrame, epochs: int = 50) -> bool:
        """Train LSTM model"""
        if not DL_AVAILABLE:
            logger.warning("Deep learning not available, using demo model")
            return False
        
        try:
            X, y = self.prepare_data(df)
            if len(X) == 0:
                return False
            
            # Split data
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Initialize model
            input_size = X.shape[2]
            self.lstm_model = LSTMModel(input_size=input_size, hidden_size=50, 
                                      num_layers=2, output_size=1)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # Training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
            
            self.lstm_model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.lstm_model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {loss.item():.6f}")
            
            self.models_trained = True
            logger.info("LSTM model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False
    
    def train_transformer_model(self, df: pd.DataFrame, epochs: int = 50) -> bool:
        """Train Transformer model"""
        if not DL_AVAILABLE:
            logger.warning("Deep learning not available, using demo model")
            return False
        
        try:
            X, y = self.prepare_data(df)
            if len(X) == 0:
                return False
            
            # Split data
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Initialize model
            input_size = X.shape[2]
            self.transformer_model = TransformerModel(input_size=input_size, d_model=64, 
                                                    nhead=8, num_layers=2, output_size=1)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            
            # Training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.transformer_model.parameters(), lr=0.001)
            
            self.transformer_model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.transformer_model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"Transformer Epoch {epoch}, Loss: {loss.item():.6f}")
            
            logger.info("Transformer model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return False
    
    def predict_price(self, df: pd.DataFrame, model_type: str = 'lstm') -> Dict:
        """Predict price using trained models"""
        try:
            if not self.models_trained:
                return self._demo_prediction(df)
            
            X, _ = self.prepare_data(df)
            if len(X) == 0:
                return self._demo_prediction(df)
            
            # Get latest sequence
            latest_sequence = X[-1:]
            
            if model_type == 'lstm' and self.lstm_model:
                self.lstm_model.eval()
                with torch.no_grad():
                    prediction = self.lstm_model(torch.FloatTensor(latest_sequence))
                    predicted_price = self.scaler.inverse_transform([[prediction.item(), 0, 0, 0, 0]])[0, 0]
            elif model_type == 'transformer' and self.transformer_model:
                self.transformer_model.eval()
                with torch.no_grad():
                    prediction = self.transformer_model(torch.FloatTensor(latest_sequence))
                    predicted_price = self.scaler.inverse_transform([[prediction.item(), 0, 0, 0, 0]])[0, 0]
            else:
                return self._demo_prediction(df)
            
            current_price = df['Close'].iloc[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            return {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change_percent': price_change,
                'model_type': model_type,
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return self._demo_prediction(df)
    
    def _demo_prediction(self, df: pd.DataFrame) -> Dict:
        """Demo prediction when models are not available"""
        current_price = df['Close'].iloc[-1]
        # Simple moving average prediction
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        predicted_price = sma_20 * 1.02  # 2% above SMA
        
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_percent': price_change,
            'model_type': 'demo',
            'confidence': 0.5
        }
    

    
    def get_trading_signal(self, df: pd.DataFrame) -> Dict:
        """Get trading signal based on technical analysis"""
        try:
            # Get price prediction
            prediction = self.predict_price(df)
            
            # Technical indicators
            rsi = self._calculate_rsi(df)
            macd = self._calculate_macd(df)
            
            # Combine signals
            signal_strength = 0
            
            # Price prediction signal
            if prediction['price_change_percent'] > 2:
                signal_strength += 1
            elif prediction['price_change_percent'] < -2:
                signal_strength -= 1
            
            # RSI signal
            if rsi < 30:
                signal_strength += 1
            elif rsi > 70:
                signal_strength -= 1
            
            # MACD signal
            if macd > 0:
                signal_strength += 0.5
            else:
                signal_strength -= 0.5
            
            # Determine final signal
            if signal_strength >= 1.5:
                signal = 'STRONG_BUY'
                confidence = min(0.9, 0.6 + abs(signal_strength) * 0.1)
            elif signal_strength >= 0.5:
                signal = 'BUY'
                confidence = min(0.8, 0.5 + abs(signal_strength) * 0.1)
            elif signal_strength <= -1.5:
                signal = 'STRONG_SELL'
                confidence = min(0.9, 0.6 + abs(signal_strength) * 0.1)
            elif signal_strength <= -0.5:
                signal = 'SELL'
                confidence = min(0.8, 0.5 + abs(signal_strength) * 0.1)
            else:
                signal = 'HOLD'
                confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'price_prediction': prediction,
                'technical_indicators': {
                    'rsi': rsi,
                    'macd': macd
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trading signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'signal_strength': 0,
                'error': str(e)
            }
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, df: pd.DataFrame) -> float:
        """Calculate MACD"""
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.iloc[-1]