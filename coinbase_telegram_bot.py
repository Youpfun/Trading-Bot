import os
import json
import requests
import logging
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import websocket
import schedule
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("adaptive_trading_bot.log", encoding='utf-8', errors='ignore'),
        logging.StreamHandler()
    ]
)

class AdaptiveTradingBot:
    def __init__(self, initial_capital: float):
        load_dotenv()
        
        # Configuration Telegram
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.telegram_token or not self.chat_id:
            logging.error("Configuration Telegram manquante")
            raise ValueError("Configuration Telegram manquante")
        
        # CAPITAL ET STRATÉGIE ADAPTATIVE
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy_config = self.get_adaptive_strategy_config(initial_capital)
        
        # FRAIS DE TRADING - AJOUTÉ
        self.maker_fee = 0.005  # 0.5% frais maker Coinbase Pro
        self.taker_fee = 0.005  # 0.5% frais taker Coinbase Pro
        self.total_fees_paid = 0.0  # Suivi des frais totaux
        
        # Configuration dynamique basée sur le capital
        self.update_strategy_parameters()
        
        # Tracking général
        self.daily_trades_count = 0
        self.last_trade_reset = datetime.now().date()
        self.multi_timeframe_data = {}
        self.active_positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Bot Telegram
        try:
            self.bot = Bot(token=self.telegram_token)
            logging.info(f"✅ Bot adaptatif initialisé - Capital: {initial_capital}€")
        except Exception as e:
            logging.error(f"Erreur Telegram: {e}")
            raise

    def get_adaptive_strategy_config(self, capital: float) -> Dict:
        """Retourne la configuration adaptée au capital - RISQUES AUGMENTÉS"""
        
        if capital <= 100:
            # STRATÉGIE ULTRA-CONSERVATRICE (≤ 100€) - RISQUE AUGMENTÉ
            return {
                "strategy_name": "ULTRA_CONSERVATIVE",
                "risk_level": "LOW",  # Changé de MINIMAL à LOW
                "max_cryptos": 2,
                "cryptos": {
                    "BTC-USD": {
                        "name": "Bitcoin",
                        "allocation_pct": 60,
                        "min_movement": 0.3,
                        "scalp_target": 0.8,
                        "swing_target": 2.5,
                        "stop_loss": 0.6,
                        "priority": 1
                    },
                    "ETH-USD": {
                        "name": "Ethereum",
                        "allocation_pct": 40,
                        "min_movement": 0.5,
                        "scalp_target": 1.2,
                        "swing_target": 3.5,
                        "stop_loss": 0.8,
                        "priority": 2
                    }
                },
                "risk_per_trade": 0.03,  # AUGMENTÉ: 1.5% → 3%
                "max_daily_trades": 6,   # AUGMENTÉ: 4 → 6
                "min_signal_strength": 6,  # RÉDUIT: 7 → 6 (plus de signaux)
                "min_rr_ratio": 1.8,      # RÉDUIT: 2.0 → 1.8
                "timeframes": ["5m", "15m", "1h"],
                "analysis_frequency": 8,  # AUGMENTÉ: 10min → 8min
                "max_position_time": 7200  # 2 heures
            }
            
        elif capital <= 500:
            # STRATÉGIE CONSERVATRICE (101-500€) - RISQUE EXTRÊME À 25%
            return {
                "strategy_name": "CONSERVATIVE_EXTREME",  # Nouveau nom pour différencier
                "risk_level": "EXTREME",  # Changé de MEDIUM à EXTREME
                "max_cryptos": 3,
                "cryptos": {
                    "BTC-USD": {
                        "name": "Bitcoin",
                        "allocation_pct": 40,
                        "min_movement": 0.4,
                        "scalp_target": 1.0,
                        "swing_target": 3.0,
                        "stop_loss": 0.8,
                        "priority": 1
                    },
                    "ETH-USD": {
                        "name": "Ethereum",
                        "allocation_pct": 35,
                        "min_movement": 0.6,
                        "scalp_target": 1.5,
                        "swing_target": 4.0,
                        "stop_loss": 1.0,
                        "priority": 2
                    },
                    "SOL-USD": {
                        "name": "Solana",
                        "allocation_pct": 25,
                        "min_movement": 0.8,
                        "scalp_target": 2.0,
                        "swing_target": 5.0,
                        "stop_loss": 1.2,
                        "priority": 3
                    }
                },
                "risk_per_trade": 0.25,  # RISQUE EXTRÊME: 25% par trade !
                "max_daily_trades": 15,  # AUGMENTÉ: 10 → 15 trades
                "min_signal_strength": 3,  # RÉDUIT: 5 → 3 (beaucoup plus de signaux)
                "min_rr_ratio": 1.0,      # RÉDUIT: 1.5 → 1.0 (risque/reward minimum)
                "timeframes": ["1m", "5m", "15m", "1h"],
                "analysis_frequency": 1,  # AUGMENTÉ: 3min → 1min (analyse chaque minute)
                "max_position_time": 7200  # 2 heures max
            }
            
        elif capital <= 1000:
            # STRATÉGIE ÉQUILIBRÉE (501-1000€) - RISQUE AUGMENTÉ
            return {
                "strategy_name": "BALANCED",
                "risk_level": "HIGH",  # Changé de MEDIUM à HIGH
                "max_cryptos": 4,
                "cryptos": {
                    "BTC-USD": {
                        "name": "Bitcoin",
                        "allocation_pct": 30,
                        "min_movement": 0.5,
                        "scalp_target": 1.2,
                        "swing_target": 3.5,
                        "stop_loss": 1.0,
                        "priority": 1
                    },
                    "ETH-USD": {
                        "name": "Ethereum",
                        "allocation_pct": 30,
                        "min_movement": 0.7,
                        "scalp_target": 1.8,
                        "swing_target": 4.5,
                        "stop_loss": 1.2,
                        "priority": 2
                    },
                    "SOL-USD": {
                        "name": "Solana",
                        "allocation_pct": 25,
                        "min_movement": 1.0,
                        "scalp_target": 2.5,
                        "swing_target": 6.0,
                        "stop_loss": 1.5,
                        "priority": 3
                    },
                    "AVAX-USD": {
                        "name": "Avalanche",
                        "allocation_pct": 15,
                        "min_movement": 1.2,
                        "scalp_target": 3.0,
                        "swing_target": 7.0,
                        "stop_loss": 1.8,
                        "priority": 4
                    }
                },
                "risk_per_trade": 0.05,  # AUGMENTÉ: 2.5% → 5%
                "max_daily_trades": 12,  # AUGMENTÉ: 8 → 12
                "min_signal_strength": 4,  # RÉDUIT: 5 → 4
                "min_rr_ratio": 1.3,      # RÉDUIT: 1.5 → 1.3
                "timeframes": ["1m", "5m", "15m", "1h", "4h"],
                "analysis_frequency": 2,  # AUGMENTÉ: 3min → 2min
                "max_position_time": 21600  # 6 heures
            }
            
        elif capital <= 5000:
            # STRATÉGIE AGRESSIVE (1001-5000€) - RISQUE TRÈS AUGMENTÉ
            return {
                "strategy_name": "AGGRESSIVE",
                "risk_level": "VERY_HIGH",  # Changé de HIGH à VERY_HIGH
                "max_cryptos": 6,
                "cryptos": {
                    "BTC-USD": {
                        "name": "Bitcoin",
                        "allocation_pct": 25,
                        "min_movement": 0.3,
                        "scalp_target": 1.0,
                        "swing_target": 4.0,
                        "stop_loss": 1.2,
                        "priority": 1
                    },
                    "ETH-USD": {
                        "name": "Ethereum",
                        "allocation_pct": 25,
                        "min_movement": 0.5,
                        "scalp_target": 1.5,
                        "swing_target": 5.0,
                        "stop_loss": 1.5,
                        "priority": 2
                    },
                    "SOL-USD": {
                        "name": "Solana",
                        "allocation_pct": 20,
                        "min_movement": 0.8,
                        "scalp_target": 2.0,
                        "swing_target": 6.0,
                        "stop_loss": 1.8,
                        "priority": 3
                    },
                    "AVAX-USD": {
                        "name": "Avalanche",
                        "allocation_pct": 15,
                        "min_movement": 1.0,
                        "scalp_target": 2.5,
                        "swing_target": 7.0,
                        "stop_loss": 2.0,
                        "priority": 4
                    },
                    "MATIC-USD": {
                        "name": "Polygon",
                        "allocation_pct": 10,
                        "min_movement": 1.2,
                        "scalp_target": 3.0,
                        "swing_target": 8.0,
                        "stop_loss": 2.2,
                        "priority": 5
                    },
                    "ADA-USD": {
                        "name": "Cardano",
                        "allocation_pct": 5,
                        "min_movement": 1.5,
                        "scalp_target": 3.5,
                        "swing_target": 9.0,
                        "stop_loss": 2.5,
                        "priority": 6
                    }
                },
                "risk_per_trade": 0.08,  # FORTEMENT AUGMENTÉ: 3% → 8%
                "max_daily_trades": 18,  # AUGMENTÉ: 12 → 18
                "min_signal_strength": 3,  # RÉDUIT: 4 → 3
                "min_rr_ratio": 1.2,      # RÉDUIT: 1.3 → 1.2
                "timeframes": ["1m", "5m", "15m", "1h", "4h"],
                "analysis_frequency": 1,  # AUGMENTÉ: 2min → 1min
                "max_position_time": 43200  # 12 heures
            }
            
        else:  # > 5000€
            # STRATÉGIE TRÈS AGRESSIVE (> 5000€) - RISQUE EXTRÊME MAINTENU
            return {
                "strategy_name": "VERY_AGGRESSIVE",
                "risk_level": "EXTREME",
                "max_cryptos": 15,
                "cryptos": {
                    "BTC-USD": {"name": "Bitcoin", "allocation_pct": 15, "min_movement": 0.1, "scalp_target": 0.5, "swing_target": 2.5, "stop_loss": 0.8, "priority": 1},
                    "ETH-USD": {"name": "Ethereum", "allocation_pct": 15, "min_movement": 0.2, "scalp_target": 0.8, "swing_target": 3.0, "stop_loss": 1.0, "priority": 2},
                    "SOL-USD": {"name": "Solana", "allocation_pct": 12, "min_movement": 0.3, "scalp_target": 1.2, "swing_target": 4.0, "stop_loss": 1.2, "priority": 3},
                    "AVAX-USD": {"name": "Avalanche", "allocation_pct": 10, "min_movement": 0.5, "scalp_target": 1.5, "swing_target": 5.0, "stop_loss": 1.5, "priority": 4},
                    "MATIC-USD": {"name": "Polygon", "allocation_pct": 8, "min_movement": 0.8, "scalp_target": 2.0, "swing_target": 6.0, "stop_loss": 1.8, "priority": 5},
                    "ADA-USD": {"name": "Cardano", "allocation_pct": 7, "min_movement": 1.0, "scalp_target": 2.5, "swing_target": 7.0, "stop_loss": 2.0, "priority": 6},
                    "DOT-USD": {"name": "Polkadot", "allocation_pct": 6, "min_movement": 1.2, "scalp_target": 3.0, "swing_target": 8.0, "stop_loss": 2.2, "priority": 7},
                    "LINK-USD": {"name": "Chainlink", "allocation_pct": 5, "min_movement": 1.5, "scalp_target": 3.5, "swing_target": 10.0, "stop_loss": 2.5, "priority": 8},
                    "UNI-USD": {"name": "Uniswap", "allocation_pct": 5, "min_movement": 1.8, "scalp_target": 4.0, "swing_target": 12.0, "stop_loss": 3.0, "priority": 9},
                    "ALGO-USD": {"name": "Algorand", "allocation_pct": 4, "min_movement": 2.0, "scalp_target": 4.5, "swing_target": 15.0, "stop_loss": 3.5, "priority": 10},
                    "XRP-USD": {"name": "Ripple", "allocation_pct": 4, "min_movement": 2.2, "scalp_target": 5.0, "swing_target": 18.0, "stop_loss": 4.0, "priority": 11},
                    "ATOM-USD": {"name": "Cosmos", "allocation_pct": 3, "min_movement": 2.5, "scalp_target": 5.5, "swing_target": 20.0, "stop_loss": 4.5, "priority": 12},
                    "ICP-USD": {"name": "Internet Computer", "allocation_pct": 2, "min_movement": 3.0, "scalp_target": 6.0, "swing_target": 25.0, "stop_loss": 5.0, "priority": 13},
                    "FTM-USD": {"name": "Fantom", "allocation_pct": 2, "min_movement": 3.5, "scalp_target": 7.0, "swing_target": 30.0, "stop_loss": 6.0, "priority": 14},
                    "NEAR-USD": {"name": "Near Protocol", "allocation_pct": 2, "min_movement": 4.0, "scalp_target": 8.0, "swing_target": 35.0, "stop_loss": 7.0, "priority": 15}
                },
                "risk_per_trade": 0.25,  # 25% DE RISQUE PAR TRADE - EXTRÊME MAINTENU
                "max_daily_trades": 50,
                "min_signal_strength": 2,
                "min_rr_ratio": 1.0,
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "analysis_frequency": 0.5,  # Analyse toutes les 30 secondes
                "max_position_time": 172800  # 48 heures max
            }

    def update_strategy_parameters(self):
        """Met à jour les paramètres selon la stratégie adaptative"""
        config = self.strategy_config
        
        # Calcul des allocations en euros
        self.specialized_cryptos = {}
        for symbol, crypto_data in config["cryptos"].items():
            allocation_amount = (self.current_capital * crypto_data["allocation_pct"]) / 100
            self.specialized_cryptos[symbol] = {
                **crypto_data,
                "capital_allocation": allocation_amount
            }
        
        # Paramètres globaux
        self.max_daily_trades = config["max_daily_trades"]
        self.max_risk_per_trade = config["risk_per_trade"]
        self.min_signal_strength = config["min_signal_strength"]
        self.min_rr_ratio = config["min_rr_ratio"]
        self.analysis_frequency = config["analysis_frequency"]
        self.max_position_time = config["max_position_time"]
        
        # Timeframes selon la stratégie - CORRECTION: Supprimer 30s non supporté par Coinbase
        valid_granularities = [60, 300, 900, 3600, 21600, 86400]  # 1m, 5m, 15m, 1h, 6h, 1d
        timeframe_mapping = {
            "1m": 60, "5m": 300, "15m": 900, 
            "1h": 3600, "4h": 14400, "6h": 21600, "1d": 86400
        }
        
        # Filtrer les timeframes valides
        self.timeframes = {}
        for tf in config["timeframes"]:
            if tf in timeframe_mapping and timeframe_mapping[tf] in valid_granularities:
                self.timeframes[tf] = timeframe_mapping[tf]
            elif tf == "30s":
                # Remplacer 30s par 1m pour mode EXTRÊME
                self.timeframes["1m"] = 60
                logging.warning("⚠️ 30s non supporté par Coinbase, utilisation de 1m")
        
        # S'assurer qu'on a au moins 1m pour les stratégies agressives
        if not self.timeframes:
            self.timeframes = {"1m": 60, "5m": 300, "15m": 900}
        
        logging.info(f"🔄 Stratégie mise à jour: {config['strategy_name']} - {len(self.specialized_cryptos)} cryptos - Risque: {config['risk_per_trade']*100}%")
        logging.info(f"📊 Timeframes: {list(self.timeframes.keys())}")

    def send_message(self, message: str):
        """Envoie un message Telegram avec gestion d'erreurs d'encodage et retry"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Nettoyer le message des caractères problématiques
                cleaned_message = message.encode('utf-8', errors='ignore').decode('utf-8')
                
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                data = {
                    'chat_id': self.chat_id,
                    'text': cleaned_message,
                    'parse_mode': 'HTML'
                }
                
                response = requests.post(url, json=data, timeout=15)  # Timeout augmenté
                if response.status_code == 200:
                    logging.info(f"Message envoyé: {cleaned_message[:50]}...")
                    return True
                else:
                    logging.warning(f"Erreur Telegram (tentative {attempt+1}): {response.status_code}")
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logging.warning(f"Erreur réseau Telegram (tentative {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Délai exponentiel
                    continue
                    
            except Exception as e:
                logging.error(f"Erreur envoi message (tentative {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        # Si tous les essais ont échoué, essayer message simplifié
        try:
            simple_message = f"Bot actif - {datetime.now().strftime('%H:%M')}"
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': simple_message
            }
            requests.post(url, json=data, timeout=10)
            logging.info("Message simplifié envoyé après échec")
            return False
        except:
            logging.error("Impossible d'envoyer même un message simplifié")
            return False

    def get_multi_timeframe_data(self, product_id):
        """Récupère les données sur plusieurs timeframes selon la stratégie - CORRIGÉE"""
        try:
            all_data = {}
            
            for tf_name, granularity in self.timeframes.items():
                try:
                    # Adapter le nombre de périodes selon le timeframe et la stratégie
                    if tf_name == "1m":
                        periods = 300 if self.strategy_config["strategy_name"] == "VERY_AGGRESSIVE" else 200
                    elif tf_name == "5m":
                        periods = 200 if self.strategy_config["strategy_name"] in ["AGRESSIVE", "VERY_AGGRESSIVE", "CONSERVATIVE_EXTREME"] else 150
                    elif tf_name == "15m":
                        periods = 150
                    elif tf_name == "1h":
                        periods = 168
                    elif tf_name == "4h":
                        periods = 120
                    elif tf_name == "6h":
                        periods = 100
                    else:  # 1d
                        periods = 90
                    
                    end_time = datetime.now()
                    start_time = end_time - timedelta(seconds=granularity * periods)
                    
                    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
                    params = {
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat(),
                        'granularity': granularity
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
                            df['time'] = pd.to_datetime(df['time'], unit='s')
                            df = df.sort_values('time').reset_index(drop=True)
                            
                            for col in ['low', 'high', 'open', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col])
                            
                            # Ajouter les indicateurs techniques adaptés
                            df = self.add_adaptive_technical_indicators(df)
                            if len(df) > 0:  # Vérifier que les données sont valides
                                all_data[tf_name] = df
                    else:
                        logging.warning(f"⚠️ Erreur API {product_id} {tf_name}: {response.status_code}")
                    
                    # Pause entre les requêtes
                    time.sleep(0.1 if self.strategy_config["strategy_name"] == "VERY_AGGRESSIVE" else 0.2)
                    
                except Exception as e:
                    logging.error(f"Erreur timeframe {tf_name} pour {product_id}: {e}")
                    continue
            
            return all_data
            
        except Exception as e:
            logging.error(f"Erreur données multi-timeframe {product_id}: {e}")
            return {}

    def add_adaptive_technical_indicators(self, df):
        """Ajoute les indicateurs techniques adaptés à la stratégie - CORRIGÉE"""
        try:
            if len(df) < 50:
                return df
            
            strategy_name = self.strategy_config["strategy_name"]
            
            # EMAs adaptées selon la stratégie
            if strategy_name == "ULTRA_CONSERVATIVE":
                # EMAs plus longues pour stabilité
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()
            elif strategy_name in ["AGRESSIVE", "VERY_AGRESSIVE"]:
                # EMAs plus courtes pour réactivité
                df['ema_5'] = df['close'].ewm(span=5).mean()
                df['ema_13'] = df['close'].ewm(span=13).mean()
                df['ema_21'] = df['close'].ewm(span=21).mean()
            else:
                # EMAs standards pour CONSERVATIVE, CONSERVATIVE_EXTREME, BALANCED
                df['ema_9'] = df['close'].ewm(span=9).mean()
                df['ema_21'] = df['close'].ewm(span=21).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # RSI adapté
            rsi_period = 21 if strategy_name == "ULTRA_CONSERVATIVE" else 14
            if strategy_name in ["VERY_AGGRESSIVE", "CONSERVATIVE_EXTREME"]:
                rsi_period = 7
            df['rsi'] = self.calculate_rsi(df['close'], rsi_period)
            
            # MACD adapté
            if strategy_name == "ULTRA_CONSERVATIVE":
                df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'], 12, 26, 9)
            elif strategy_name in ["AGRESSIVE", "VERY_AGGRESSIVE", "CONSERVATIVE_EXTREME"]:
                df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'], 5, 13, 3)
            else:
                df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'], 8, 21, 5)
            
            # Bollinger Bands adaptées
            bb_period = 25 if strategy_name == "ULTRA_CONSERVATIVE" else 20
            bb_std = 2.5 if strategy_name == "ULTRA_CONSERVATIVE" else 2.0
            if strategy_name in ["VERY_AGGRESSIVE", "CONSERVATIVE_EXTREME"]:
                bb_period, bb_std = 15, 1.5
            
            df['bb_upper'], df['bb_lower'], df['bb_middle'] = self.calculate_bollinger_bands(df['close'], bb_period, bb_std)
            
            # Volume analysis
            volume_window = 20 if strategy_name == "ULTRA_CONSERVATIVE" else 10
            df['volume_sma'] = df['volume'].rolling(window=volume_window).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Support/Resistance
            sr_window = 20 if strategy_name == "ULTRA_CONSERVATIVE" else 10
            if strategy_name in ["VERY_AGGRESSIVE", "CONSERVATIVE_EXTREME"]:
                sr_window = 5
            df['support'], df['resistance'] = self.calculate_support_resistance(df, sr_window)
            
            # ATR et volatilité
            atr_period = 20 if strategy_name == "ULTRA_CONSERVATIVE" else 14
            df['atr'] = self.calculate_atr(df, atr_period)
            df['volatility'] = df['close'].pct_change().rolling(window=atr_period).std()
            
            # Indicateurs avancés pour stratégies agressives
            if strategy_name in ["AGRESSIVE", "VERY_AGGRESSIVE", "CONSERVATIVE_EXTREME"]:
                # Stochastic
                df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df)
                # Williams %R
                df['williams_r'] = self.calculate_williams_r(df)
                # Momentum
                df['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            return df.dropna()
            
        except Exception as e:
            logging.error(f"Erreur indicateurs adaptatifs: {e}")
            return df

    def calculate_rsi(self, prices, period=14):
        """RSI calculation - CORRIGÉE avec gestion division par zéro"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Éviter division par zéro
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Remplacer les valeurs infinies
            rsi = rsi.replace([np.inf, -np.inf], np.nan)
            
            return rsi.fillna(50)  # Valeur neutre par défaut
            
        except Exception as e:
            logging.error(f"Erreur calcul RSI: {e}")
            return pd.Series(index=prices.index, data=50)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD calculation"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Bollinger Bands calculation - CORRIGÉE"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            # Éviter std = 0 et NaN
            std = std.replace(0, np.nan)
            std = std.fillna(std.mean())
            std = std.fillna(0.001)  # Fallback si tout est NaN
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return upper, lower, sma
            
        except Exception as e:
            logging.error(f"Erreur Bollinger Bands: {e}")
            return prices, prices, prices

    def calculate_support_resistance(self, df, window=20):
        """Support/Résistance - CORRIGÉE"""
        try:
            if len(df) < window:
                return df['low'], df['high']
                
            highs = df['high'].rolling(window=window).max()
            lows = df['low'].rolling(window=window).min()
            
            # Gérer les NaN
            highs = highs.fillna(df['high'])
            lows = lows.fillna(df['low'])
            
            return lows, highs
        except Exception as e:
            logging.error(f"Erreur support/résistance: {e}")
            return df['low'], df['high']

    def calculate_atr(self, df, period=14):
        """Average True Range - CORRIGÉE"""
        try:
            if len(df) < period:
                return df['high'] - df['low']
                
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            # Gérer les NaN
            atr = atr.fillna(high_low)
            
            return atr
        except Exception as e:
            logging.error(f"Erreur ATR: {e}")
            return df['high'] - df['low']

    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Stochastic Oscillator - CORRIGÉE"""
        try:
            if len(df) < k_period:
                return pd.Series(index=df.index, data=50), pd.Series(index=df.index, data=50)
                
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            # Éviter division par zéro
            denominator = high_max - low_min
            denominator = denominator.replace(0, np.nan).fillna(0.001)
            
            k_percent = 100 * ((df['close'] - low_min) / denominator)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            # Gérer les valeurs aberrantes
            k_percent = k_percent.clip(0, 100).fillna(50)
            d_percent = d_percent.clip(0, 100).fillna(50)
            
            return k_percent, d_percent
        except Exception as e:
            logging.error(f"Erreur Stochastic: {e}")
            return pd.Series(index=df.index, data=50), pd.Series(index=df.index, data=50)

    def calculate_williams_r(self, df, period=14):
        """Williams %R - CORRIGÉE"""
        try:
            if len(df) < period:
                return pd.Series(index=df.index, data=-50)
                
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            
            # Éviter division par zéro
            denominator = high_max - low_min
            denominator = denominator.replace(0, np.nan).fillna(0.001)
            
            williams_r = -100 * ((high_max - df['close']) / denominator)
            
            # Gérer les valeurs aberrantes
            williams_r = williams_r.clip(-100, 0).fillna(-50)
            
            return williams_r
        except Exception as e:
            logging.error(f"Erreur Williams R: {e}")
            return pd.Series(index=df.index, data=-50)

    def run_adaptive_analysis(self):
        """Lance l'analyse adaptée à la stratégie - CORRIGÉE"""
        try:
            # Reset compteurs si nouveau jour
            self.reset_daily_counters()
            
            strategy_name = self.strategy_config["strategy_name"]
            logging.info(f"🔄 Analyse {strategy_name} en cours...")
            
            trades_executed = 0
            
            # Analyser les cryptos selon l'allocation
            for product_id, crypto_config in self.specialized_cryptos.items():
                try:
                    # Vérifier si on a atteint la limite quotidienne
                    if self.daily_trades_count >= self.max_daily_trades:
                        logging.info(f"Limite quotidienne atteinte: {self.max_daily_trades} trades")
                        break
                    
                    analysis = self.analyze_adaptive_signal(product_id, crypto_config)
                    if analysis and analysis['signal_type'] in ['BUY', 'SELL']:
                        # Vérifier que le signal est assez fort
                        if analysis['signal_strength'] >= self.min_signal_strength:
                            self.execute_adaptive_trade(analysis, crypto_config)
                            trades_executed += 1
                        else:
                            logging.info(f"Signal {product_id} trop faible: {analysis['signal_strength']}/{self.min_signal_strength}")
                    
                    # Pause adaptée à la stratégie
                    pause_time = 0.5 if strategy_name == "VERY_AGGRESSIVE" else 1
                    time.sleep(pause_time)
                    
                except Exception as e:
                    logging.error(f"Erreur analyse {product_id}: {e}")
                    continue
            
            # Vérifier les conditions de sortie
            self.check_adaptive_exit_conditions()
            
            # Message de fin d'analyse (seulement si des trades exécutés ou positions actives)
            if trades_executed > 0 or len(self.active_positions) > 0:
                current_capital = self.initial_capital + self.total_pnl
                performance_pct = ((current_capital - self.initial_capital) / self.initial_capital) * 100
                
                # Affichage de la fréquence corrigée
                freq_display = f"{self.analysis_frequency}min" if self.analysis_frequency >= 1 else f"{int(self.analysis_frequency*60)}s"
                
                message = f"✅ <b>ANALYSE {strategy_name}</b>\n"
                message += f"🎯 {trades_executed} nouveau(x) trade(s)\n"
                message += f"📊 {len(self.active_positions)} position(s) active(s)\n"
                message += f"💰 Capital: {current_capital:.2f}€ ({performance_pct:+.1f}%)\n"
                message += f"⏰ Prochaine analyse dans {freq_display}"
                
                self.send_message(message)
            
            # Vérifier si on doit adapter la stratégie (capital a changé significativement)
            self.check_strategy_adaptation()
            
        except Exception as e:
            logging.error(f"Erreur analyse adaptative: {e}")

    def analyze_adaptive_signal(self, product_id, crypto_config):
        """Analyse adaptative selon la stratégie - CORRIGÉE"""
        try:
            tf_data = self.get_multi_timeframe_data(product_id)
            if not tf_data:
                return None
            
            self.multi_timeframe_data[product_id] = tf_data
            
            strategy_name = self.strategy_config["strategy_name"]
            
            if strategy_name == "ULTRA_CONSERVATIVE":
                return self.analyze_ultra_conservatrice(product_id, crypto_config, tf_data)
            elif strategy_name == "CONSERVATIVE":
                return self.analyze_conservative(product_id, crypto_config, tf_data)
            elif strategy_name == "CONSERVATIVE_EXTREME":  # CORRIGÉ
                return self.analyze_conservative_extreme(product_id, crypto_config, tf_data)
            elif strategy_name == "BALANCED":
                return self.analyze_balanced(product_id, crypto_config, tf_data)
            elif strategy_name == "AGGRESSIVE":  # CORRIGÉ: "AGRESSIVE" → "AGGRESSIVE"
                return self.analyze_aggressive(product_id, crypto_config, tf_data)
            elif strategy_name == "VERY_AGGRESSIVE":  # AJOUTÉ
                return self.analyze_very_aggressive(product_id, crypto_config, tf_data)
            else:
                logging.warning(f"Stratégie inconnue: {strategy_name}")
                return None
                
        except Exception as e:
            logging.error(f"Erreur analyse adaptative {product_id}: {e}")
            return None
        
    def check_strategy_adaptation(self):
        """Vérifie si la stratégie doit être adaptée selon l'évolution du capital"""
        try:
            current_capital = self.initial_capital + self.total_pnl
            capital_change_pct = ((current_capital - self.current_capital) / self.current_capital) * 100
            
            # Si le capital a changé de plus de 20%, on peut adapter la stratégie
            if abs(capital_change_pct) > 20:
                new_config = self.get_adaptive_strategy_config(current_capital)
                
                if new_config["strategy_name"] != self.strategy_config["strategy_name"]:
                    old_strategy = self.strategy_config["strategy_name"]
                    
                    self.current_capital = current_capital
                    self.strategy_config = new_config
                    self.update_strategy_parameters()
                    
                    message = f"🔄 <b>ADAPTATION STRATÉGIQUE</b>\n"
                    message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    message += f"💰 Capital: {current_capital:.2f}€\n"
                    message += f"📈 Evolution: {capital_change_pct:+.1f}%\n\n"
                    message += f"🔄 <b>CHANGEMENT DE STRATÉGIE:</b>\n"
                    message += f"❌ Ancienne: {old_strategy}\n"
                    message += f"✅ Nouvelle: {new_config['strategy_name']}\n\n"
                    message += f"🎯 Nouveau paramétrage:\n"
                    message += f"• Cryptos: {len(new_config['cryptos'])}\n"
                    message += f"• Risque: {new_config['risk_per_trade']*100:.1f}%\n"
                    message += f"• Trades max: {new_config['max_daily_trades']}/jour\n"
                    message += f"• Analyses: {new_config['analysis_frequency']}min\n\n"
                    message += f"🤖 Bot adapté automatiquement!"
                    
                    self.send_message(message)
                    logging.info(f"🔄 Stratégie adaptée: {old_strategy} → {new_config['strategy_name']}")
                    
        except Exception as e:
            logging.error(f"Erreur adaptation stratégie: {e}")

    def run(self):
        """Lance le bot adaptatif - CORRIGÉE"""
        try:
            # Message d'initialisation avec stratégie personnalisée
            self.send_strategy_initialization_message()
            
            # Analyse initiale
            self.run_adaptive_analysis()
            
            # Programmation adaptée selon la fréquence
            if self.analysis_frequency >= 1:
                # Fréquence en minutes
                schedule.every(int(self.analysis_frequency)).minutes.do(self.run_adaptive_analysis)
            else:
                # Fréquence en secondes (pour mode EXTRÊME)
                seconds = int(self.analysis_frequency * 60)
                schedule.every(seconds).seconds.do(self.run_adaptive_analysis)
            
            schedule.every().day.at("20:00").do(self.send_daily_adaptive_summary)
            
            # Boucle principale
            logging.info(f"🎯 Bot adaptatif opérationnel - Stratégie: {self.strategy_config['strategy_name']}")
            while True:
                schedule.run_pending()
                sleep_time = 5 if self.strategy_config["strategy_name"] == "VERY_AGGRESSIVE" else 15
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logging.info("Arrêt bot adaptatif")
            final_capital = self.initial_capital + self.total_pnl
            final_performance = ((final_capital - self.initial_capital) / self.initial_capital) * 100
            
            self.send_message(f"🛑 <b>BOT ADAPTATIF ARRÊTÉ</b>\n💰 Capital final: {final_capital:.2f}€\n📊 Performance: {final_performance:+.1f}%\n🎯 Stratégie: {self.strategy_config['strategy_name']}")
        except Exception as e:
            logging.error(f"Erreur critique: {e}")
            self.send_message(f"❌ Erreur critique: {e}")

    def send_daily_adaptive_summary(self):
        """Envoie le résumé quotidien adaptatif avec frais RÉELS"""
        try:
            current_capital = self.initial_capital + self.total_pnl
            performance_pct = ((current_capital - self.initial_capital) / self.initial_capital) * 100
            
            # Statistiques RÉELLES des trades avec frais
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t.get('net_pnl_amount', 0) > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Moyenne des gains/pertes NET (après frais)
            if total_trades > 0:
                net_profits = [t.get('net_pnl_amount', 0) for t in self.trade_history if t.get('net_pnl_amount', 0) > 0]
                net_losses = [t.get('net_pnl_amount', 0) for t in self.trade_history if t.get('net_pnl_amount', 0) < 0]
                
                avg_profit = np.mean(net_profits) if net_profits else 0
                avg_loss = np.mean(net_losses) if net_losses else 0
                avg_duration = np.mean([
                    (t.get('exit_time', datetime.now()) - t.get('entry_time', datetime.now())).total_seconds() / 3600 
                    for t in self.trade_history if 'exit_time' in t and 'entry_time' in t
                ])
            else:
                avg_profit = avg_loss = avg_duration = 0
            
            message = f"📊 <b>RÉSUMÉ QUOTIDIEN - AVEC FRAIS RÉELS</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📅 {datetime.now().strftime('%d/%m/%Y')}\n\n"
            
            message += f"💰 <b>PERFORMANCE RÉELLE:</b>\n"
            message += f"💵 Capital initial: {self.initial_capital:.2f}€\n"
            message += f"💎 Capital actuel: <b>{current_capital:.2f}€</b>\n"
            message += f"📈 Performance NET: <b>{performance_pct:+.1f}%</b>\n"
            message += f"💵 P&L aujourd'hui: {self.daily_pnl:+.2f}€\n"
            message += f"💵 P&L total NET: {self.total_pnl:+.2f}€\n"
            message += f"💸 Frais totaux payés: {self.total_fees_paid:.2f}€\n\n"
            
            message += f"📊 <b>STATISTIQUES AVEC FRAIS:</b>\n"
            message += f"🔢 Trades total: {total_trades}\n"
            message += f"✅ Trades gagnants: {winning_trades}\n"
            message += f"📈 Taux de réussite: {win_rate:.1f}%\n"
            if total_trades > 0:
                message += f"📈 Gain moyen NET: +{avg_profit:.2f}€\n"
                message += f"📉 Perte moyenne NET: {avg_loss:.2f}€\n"
                message += f"⏰ Durée moyenne: {avg_duration:.1f}h\n"
                message += f"💸 Frais moyen/trade: {self.total_fees_paid/total_trades:.2f}€\n"
            message += f"📊 Positions actives: {len(self.active_positions)}\n\n"
            
            # Liste des positions actives avec P&L en temps réel (avant frais)
            if self.active_positions:
                message += f"🔄 <b>POSITIONS ACTIVES (avant frais sortie):</b>\n"
                for product_id, position in self.active_positions.items():
                    current_price = self.get_current_price(product_id)
                    if current_price:
                        entry_price = position['entry_price']
                        if position['signal_type'] == 'BUY':
                            unrealized_pnl = ((current_price - entry_price) / entry_price) * 100
                        else:
                            unrealized_pnl = ((entry_price - current_price) / entry_price) * 100
                        
                        # Estimer P&L net après frais de sortie
                        estimated_exit_fee = position['risk_amount'] * self.taker_fee
                        total_fees = position['entry_fee'] + estimated_exit_fee
                        net_pnl_estimate = (position['risk_amount'] * unrealized_pnl / 100) - total_fees
                        
                        duration = datetime.now() - position['entry_time']
                        pnl_emoji = "📈" if net_pnl_estimate > 0 else "📉"
                        
                        message += f"• {product_id}: {pnl_emoji} {unrealized_pnl:+.1f}% (NET: {net_pnl_estimate:+.1f}€) - {duration}\n"
                message += "\n"
            
            message += f"🎯 <b>STRATÉGIE:</b> {self.strategy_config['strategy_name']}\n"
            message += f"💸 Risque: {self.strategy_config['risk_per_trade']*100:.1f}%/trade\n"
            message += f"💸 Frais: 1.0% par trade (0.5% entrée + 0.5% sortie)\n\n"
            
            message += f"🤖 Bot adaptatif avec frais RÉELS calculés"
            
            self.send_message(message)
            
            # Reset compteurs quotidiens
            self.daily_pnl = 0.0
            self.daily_trades_count = 0
            self.last_trade_reset = datetime.now().date()
            
        except Exception as e:
            logging.error(f"Erreur résumé quotidien: {e}")

    def send_strategy_initialization_message(self):
        """Envoie le message d'initialisation avec la stratégie adaptée - RISQUES AUGMENTÉS"""
        try:
            config = self.strategy_config
            
            # Emoji selon le niveau de risque
            risk_emoji = {
                "MINIMAL": "🛡️",
                "LOW": "🟢", 
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "VERY_HIGH": "🔴",
                "EXTREME": "💀"
            }
            
            message = f"🚀 <b>BOT ADAPTIF DÉMARRÉ - RISQUES AUGMENTÉS</b> 🚀\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"💰 <b>Capital initial: {self.initial_capital:.0f}€</b>\n"
            message += f"{risk_emoji.get(config['risk_level'], '❓')} <b>Stratégie: {config['strategy_name']}</b>\n"
            message += f"⚖️ Niveau de risque: {config['risk_level']} ⬆️\n\n"
            
            # Avertissement spécial selon le capital
            if self.initial_capital <= 100:
                message += f"⚠️ <b>PETIT CAPITAL - RISQUE DOUBLÉ:</b>\n"
                message += f"🔥 Risque: <b>3% par trade</b> (au lieu de 1.5%)\n"
                message += f"⚡ Analyses: <b>toutes les 8 minutes</b>\n"
                message += f"🎯 Plus de trades possibles!\n\n"
            elif self.initial_capital <= 500:
                message += f"💀 <b>CAPITAL MOYEN - RISQUE EXTRÊME 25%:</b>\n"
                message += f"🔥 Risque: <b>25% par trade</b> (MAXIMUM ABSOLU!)\n"
                message += f"⚡ Analyses: <b>toutes les minutes</b>\n"
                message += f"🎯 15 trades max/jour!\n"
                message += f"⚠️ ATTENTION: RISQUE DE PERTE TOTALE ÉLEVÉ!\n"
                message += f"💸 Un seul mauvais trade = -25% du capital!\n\n"
            elif self.initial_capital <= 1000:
                message += f"⚠️ <b>CAPITAL ÉLEVÉ - RISQUE DOUBLÉ:</b>\n"
                message += f"🔥 Risque: <b>5% par trade</b> (au lieu de 2.5%)\n"
                message += f"⚡ Analyses: <b>toutes les 2 minutes</b>\n"
                message += f"🎯 12 trades max/jour!\n\n"
            elif self.initial_capital <= 5000:
                message += f"⚠️ <b>GROS CAPITAL - RISQUE TRÈS AUGMENTÉ:</b>\n"
                message += f"🔥 Risque: <b>8% par trade</b> (au lieu de 3%)\n"
                message += f"⚡ Analyses: <b>toutes les minutes</b>\n"
                message += f"🎯 18 trades max/jour!\n"
                message += f"💸 Objectif: Gains rapides et agressifs!\n\n"
            else:
                message += f"💀 <b>CAPITAL TRÈS ÉLEVÉ - MODE EXTRÊME:</b>\n"
                message += f"💀 Mode EXTRÊME maintenu!\n"
                message += f"🔥 Risque: <b>25% par trade</b>\n"
                message += f"⚡ Analyses: <b>toutes les 30 secondes</b>\n"
                message += f"🎰 Objectif: Gains explosifs\n"
                message += f"💸 Risque de perte importante!\n\n"
            
            message += f"🎯 <b>CONFIGURATION ADAPTATIVE AGRESSIVE:</b>\n"
            message += f"📊 Cryptos tradées: <b>{len(config['cryptos'])}</b>\n"
            message += f"💸 Risque par trade: <b>{config['risk_per_trade']*100:.1f}%</b> ⬆️\n"
            message += f"🔢 Trades max/jour: <b>{config['max_daily_trades']}</b> ⬆️\n"
            message += f"💪 Signal minimum: <b>{config['min_signal_strength']}/9</b> ⬇️\n"
            message += f"📈 R/R minimum: <b>{config['min_rr_ratio']:.1f}:1</b> ⬇️\n"
            
            # Fréquence d'analyse adaptée
            freq_text = f"{config['analysis_frequency']}min" if config['analysis_frequency'] >= 1 else f"{int(config['analysis_frequency']*60)}s"
            message += f"⏰ Analyses: <b>toutes les {freq_text}</b> ⬆️\n\n"
            
            message += f"💼 <b>RÉPARTITION DU CAPITAL:</b>\n"
            for symbol, crypto_data in self.specialized_cryptos.items():
                crypto_name = crypto_data['name']
                allocation = crypto_data['capital_allocation']
                percentage = crypto_data['allocation_pct']
                message += f"• <b>{crypto_name}:</b> {allocation:.0f}€ ({percentage}%)\n"
            
            message += f"\n📊 <b>TIMEFRAMES UTILISÉS:</b>\n"
            tf_names = list(self.timeframes.keys())
            message += f"• {', '.join(tf_names)}\n\n"
            
            # Stratégies spécifiques selon le capital avec risques augmentés
            if self.initial_capital <= 100:
                message += f"🔥 <b>Mode Ultra-Conservateur AGRESSIF:</b>\n"
                message += f"• Priorité: Croissance accélérée\n"
                message += f"• Objectif: +5-10% par mois (au lieu de +2-5%)\n"
                message += f"• Style: Scalping agressif\n"
                message += f"• Risque doublé pour maximiser les gains!\n"
            elif self.initial_capital <= 500:
                message += f"💀 <b>Mode Conservateur EXTRÊME - 25% RISQUE:</b>\n"
                message += f"• Priorité: Gains explosifs ou pertes massives\n"
                message += f"• Objectif: +50-100% par mois (ou -100%)\n"
                message += f"• Style: Trading ultra-agressif\n"
                message += f"• RISQUE MAXIMUM: 25% par trade!\n"
                message += f"• ⚠️ PEUT PERDRE TOUT LE CAPITAL RAPIDEMENT!\n"
            elif self.initial_capital <= 1000:
                message += f"🔥 <b>Mode Équilibré TRÈS AGRESSIF:</b>\n"
                message += f"• Priorité: Gains rapides\n"
                message += f"• Objectif: +15-30% par mois (au lieu de +8-15%)\n"
                message += f"• Style: Trading haute fréquence\n"
                message += f"• Risque doublé pour performance maximale!\n"
            elif self.initial_capital <= 5000:
                message += f"🔥 <b>Mode Agressif EXTRÊME:</b>\n"
                message += f"• Priorité: Gains explosifs\n"
                message += f"• Objectif: +25-50% par mois (au lieu de +12-25%)\n"
                message += f"• Style: Trading ultra-agressif\n"
                message += f"• Risque triplé - Gains potentiels énormes!\n"
            else:
                message += f"💀 <b>Mode EXTRÊME MAINTENU:</b>\n"
                message += f"• Priorité: Gains explosifs\n"
                message += f"• Objectif: +50-200% par mois\n"
                message += f"• Style: Trading ultra-haute fréquence\n"
                message += f"• ⚠️ RISQUE MAXIMUM MAINTENU!\n"
            
            message += f"\n🔥 <b>ATTENTION: RISQUES AUGMENTÉS!</b>\n"
            message += f"💪 Plus de trades, plus de gains potentiels\n"
            message += f"⚠️ Mais aussi plus de risques\n"
            message += f"🎯 Objectif: Performance maximale\n\n"
            
            message += f"✅ <b>Bot adaptatif agressif opérationnel!</b>\n"
            message += f"🤖 Stratégie ultra-agressive selon votre capital"
            
            self.send_message(message)
            
        except Exception as e:
            logging.error(f"Erreur message initialization: {e}")

    def execute_adaptive_trade(self, analysis, crypto_config):
        """Exécute un trade adaptatif avec gestion du risque et frais"""
        try:
            product_id = analysis['product_id']
            signal_type = analysis['signal_type']
            current_price = analysis['current_price']
            
            # VÉRIFICATION SPÉCIALE pour CONSERVATIVE_EXTREME
            if self.strategy_config["strategy_name"] == "CONSERVATIVE_EXTREME":
                # Vérifier que le signal est vraiment fort
                if analysis['signal_strength'] < 4:
                    logging.info(f"Signal {product_id} trop faible pour mode EXTREME: {analysis['signal_strength']}/4")
                    return
            
            # Vérifier les limites de trades quotidiens
            if self.daily_trades_count >= self.max_daily_trades:
                logging.warning(f"Limite quotidienne de trades atteinte: {self.max_daily_trades}")
                return
            
            # Vérifier si on a déjà une position sur cette crypto
            if product_id in self.active_positions:
                logging.warning(f"Position déjà active sur {product_id}")
                return
            
            # Calcul de la taille de position selon le niveau de risque
            risk_amount = self.current_capital * self.max_risk_per_trade
            
            # Calculer les frais d'entrée
            entry_fee = risk_amount * self.taker_fee
            estimated_exit_fee = risk_amount * self.taker_fee
            total_estimated_fees = entry_fee + estimated_exit_fee
            
            # Pour mode EXTRÊME, position encore plus agressive
            if self.strategy_config["strategy_name"] == "VERY_AGGRESSIVE" and self.initial_capital > 5000:
                risk_amount *= 1.2  # 20% de plus en mode EXTRÊME
            
            position = {
                'product_id': product_id,
                'signal_type': signal_type,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'signal_strength': analysis['signal_strength'],
                'reasons': analysis['reasons'],
                'risk_amount': risk_amount,
                'position_size': risk_amount / current_price,
                'entry_fee': entry_fee,
                'estimated_total_fees': total_estimated_fees
            }
            
            self.active_positions[product_id] = position
            self.daily_trades_count += 1
            
            # Alerte avec info de risque ET frais
            emoji = "🟢" if signal_type == "BUY" else "🔴"
            action = "ACHAT" if signal_type == "BUY" else "VENTE"
            
            message = f"{emoji} <b>SIGNAL {action}</b>"
            
            # Avertissement spécial pour mode EXTRÊME
            if self.strategy_config["strategy_name"] == "VERY_AGGRESSIVE" and self.initial_capital > 5000:
                message += f" 💀\n"
                message += f"⚠️ <b>MODE EXTRÊME ACTIVÉ</b>\n"
            else:
                message += f"\n"
            
            message += f"💰 {product_id}\n"
            message += f"💵 Prix: ${current_price:.4f}\n"
            message += f"💪 Force: {analysis['signal_strength']}/9\n"
            message += f"🎯 Stratégie: {self.strategy_config['strategy_name']}\n"
            message += f"💸 Risque: {risk_amount:.2f}€ ({self.max_risk_per_trade*100:.0f}%)\n"
            message += f"📊 Taille: {position['position_size']:.6f} {product_id.split('-')[0]}\n"
            message += f"💸 Frais estimés: {total_estimated_fees:.2f}€ (1.0%)\n"
            message += f"📈 Seuil rentabilité: +{(total_estimated_fees/risk_amount)*100:.1f}%\n"
            message += f"📈 Raisons:\n"
            for reason in analysis['reasons']:
                message += f"• {reason}\n"
            
            # Avertissement spécial pour mode CONSERVATIVE_EXTREME
            if self.strategy_config["strategy_name"] == "CONSERVATIVE_EXTREME":
                message += f"⚠️ MODE 25% - Seuil rentabilité: +2.5%\n"
                message += f"💀 1% de frais - Trade rentable si gain > +2.5%\n"
            
            message += f"\n⚠️ FRAIS: {total_estimated_fees:.2f}€ par trade!"
            
            self.send_message(message)
            logging.info(f"Trade {signal_type}: {product_id} - Force: {analysis['signal_strength']} - Frais: {total_estimated_fees:.2f}€")
            
        except Exception as e:
            logging.error(f"Erreur exécution trade: {e}")

    def check_adaptive_exit_conditions(self):
        """Vérifie les conditions de sortie adaptatives - OPTIMISÉE POUR FRAIS"""
        try:
            for product_id, position in list(self.active_positions.items()):
                try:
                    current_price = self.get_current_price(product_id)
                    if current_price is None:
                        continue
                    
                    entry_price = position['entry_price']
                    signal_type = position['signal_type']
                    time_elapsed = datetime.now() - position['entry_time']
                    
                    # Calculer le P&L actuel RÉEL
                    if signal_type == 'BUY':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:  # SELL
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    should_exit = False
                    exit_reason = ""
                    strategy_name = self.strategy_config["strategy_name"]
                    crypto_config = self.specialized_cryptos.get(product_id, {})
                    
                    # Take Profit AUGMENTÉ pour CONSERVATIVE_EXTREME (25% risque)
                    if strategy_name == "CONSERVATIVE_EXTREME":
                        # Take profit plus élevé pour compenser les frais et le risque
                        if pnl_pct >= 2.5:  # 2.5% minimum (était 6.2%)
                            should_exit = True
                            exit_reason = "Take Profit 2.5% (frais compensés)"
                    elif strategy_name == "ULTRA_CONSERVATIVE":
                        if pnl_pct >= 2.0:  # 2.0% (était 2.7%)
                            should_exit = True
                            exit_reason = "Take Profit conservateur"
                    elif strategy_name == "CONSERVATIVE":
                        if pnl_pct >= 2.2:  # 2.2% (était 3.2%)
                            should_exit = True
                            exit_reason = "Take Profit"
                    elif strategy_name == "BALANCED":
                        if pnl_pct >= 3.0:
                            should_exit = True
                            exit_reason = "Take Profit équilibré"
                    elif strategy_name == "AGGRESSIVE":
                        if pnl_pct >= 4.0:
                            should_exit = True
                            exit_reason = "Take Profit agressif"
                    else:  # VERY_AGGRESSIVE
                        if pnl_pct >= 5.0:
                            should_exit = True
                            exit_reason = "Take Profit EXTRÊME"
                    
                    # Stop Loss PLUS TOLÉRANT pour mode EXTREME
                    stop_loss_pct = crypto_config.get('stop_loss', 2.0)
                    
                    if strategy_name == "CONSERVATIVE_EXTREME":
                        # Stop Loss plus tolérant car risque 25%
                        stop_loss_pct = 5.0  # 5% au lieu de 4%
                    else:
                        stop_loss_pct = stop_loss_pct + 1.2  # Compensation frais
                    
                    if pnl_pct <= -stop_loss_pct:
                        should_exit = True
                        exit_reason = f"Stop Loss -{stop_loss_pct:.1f}%"
                    
                    # Exit temporel - PLUS TOLÉRANT pour CONSERVATIVE_EXTREME
                    if strategy_name == "CONSERVATIVE_EXTREME":
                        max_time = timedelta(seconds=1800)  # 30 minutes au lieu de 2h
                    else:
                        max_time = timedelta(seconds=self.max_position_time)
                        
                    if time_elapsed > max_time:
                        should_exit = True
                        exit_reason = "Time Exit"
                    
                    # Conditions techniques - MOINS AGRESSIVES
                    if not should_exit and time_elapsed.total_seconds() > 300:  # Après 5 min seulement
                        tf_data = self.get_multi_timeframe_data(product_id)
                        if tf_data:
                            should_exit, tech_reason = self.check_technical_exit(product_id, position, tf_data, current_price)
                            if should_exit:
                                exit_reason = tech_reason
                    
                    # FERMER LA POSITION si conditions remplies
                    if should_exit:
                        self.close_real_position(product_id, pnl_pct, current_price, exit_reason)
                        
                except Exception as e:
                    logging.error(f"Erreur vérification position {product_id}: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Erreur vérification sorties: {e}")

    def get_current_price(self, product_id):
        """Récupère le prix actuel RÉEL d'une crypto"""
        try:
            url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                logging.warning(f"Erreur prix {product_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Erreur récupération prix {product_id}: {e}")
            return None

    def check_technical_exit(self, product_id, position, tf_data, current_price):
        """Vérifie les conditions de sortie techniques - PLUS TOLÉRANTE"""
        try:
            strategy_name = self.strategy_config["strategy_name"]
            signal_type = position['signal_type']
            time_elapsed = datetime.now() - position['entry_time']
            
            # Pour CONSERVATIVE_EXTREME, être plus tolérant sur les sorties techniques
            if strategy_name == "CONSERVATIVE_EXTREME":
                # Ne pas sortir dans les 5 premières minutes sauf urgence
                if time_elapsed.total_seconds() < 300:  # 5 minutes
                    return False, ""
            
            # Données 5 minutes pour exit rapide
            m5_data = tf_data.get('5m')
            if m5_data is None or len(m5_data) < 20:
                return False, ""
            
            latest = m5_data.iloc[-1]
            
            # RSI exit conditions - PLUS STRICTES pour éviter sorties prématurées
            if 'rsi' in latest and not pd.isna(latest['rsi']):
                # Seuils plus extrêmes pour mode 25% risque
                if signal_type == 'BUY' and latest['rsi'] > 85:  # 85 au lieu de 75
                    return True, "RSI surachat extrême (85+)"
                elif signal_type == 'SELL' and latest['rsi'] < 15:  # 15 au lieu de 25
                    return True, "RSI survente extrême (15-)"
            
            # Volume faible - PLUS TOLÉRANT pour mode EXTREME
            if 'volume_ratio' in latest:
                if strategy_name == "CONSERVATIVE_EXTREME":
                    # Volume très faible seulement
                    if latest['volume_ratio'] < 0.3:  # 0.3 au lieu de 0.5
                        return True, "Volume très faible"
                else:
                    if latest['volume_ratio'] < 0.5:
                        return True, "Volume faible"
            
            # MACD divergence - SEULEMENT pour signaux TRÈS forts
            if all(col in latest for col in ['macd', 'macd_signal']):
                macd_diff = abs(latest['macd'] - latest['macd_signal'])
                # Ne sortir que si divergence très forte
                if macd_diff > 0.01:  # Seuil plus élevé
                    if signal_type == 'BUY' and latest['macd'] < latest['macd_signal']:
                        return True, "MACD divergence forte"
                    elif signal_type == 'SELL' and latest['macd'] > latest['macd_signal']:
                        return True, "MACD divergence forte"
            
            return False, ""
            
        except Exception as e:
            logging.error(f"Erreur exit technique {product_id}: {e}")
            return False, ""

    def close_real_position(self, product_id, real_pnl_pct, current_price, exit_reason):
        """Ferme une position avec P&L RÉEL calculé avec frais"""
        try:
            if product_id not in self.active_positions:
                return
            
            position = self.active_positions[product_id]
            
            # Calcul des frais RÉELS
            position_value = position['risk_amount']
            entry_fee = position_value * self.taker_fee  # Frais d'entrée (taker)
            exit_fee = position_value * self.taker_fee   # Frais de sortie (taker)
            total_fees = entry_fee + exit_fee
            
            # P&L BRUT (sans frais)
            gross_pnl = position['risk_amount'] * (real_pnl_pct / 100)
            
            # P&L NET (après frais)
            net_pnl = gross_pnl - total_fees
            
            # Mise à jour des totaux avec P&L NET
            self.total_pnl += net_pnl
            self.daily_pnl += net_pnl
            self.current_capital += net_pnl
            self.total_fees_paid += total_fees
            
            # Historique avec données RÉELLES incluant frais
            trade_record = {
                'product_id': product_id,
                'signal_type': position['signal_type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'exit_reason': exit_reason,
                'gross_pnl_pct': real_pnl_pct,  # P&L brut
                'gross_pnl_amount': gross_pnl,  # Montant brut
                'net_pnl_amount': net_pnl,      # Montant net (après frais)
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'total_fees': total_fees,
                'risk_amount': position['risk_amount'],
                'strategy': self.strategy_config['strategy_name'],
                'duration': datetime.now() - position['entry_time']
            }
            self.trade_history.append(trade_record)
            
            del self.active_positions[product_id]
            
            # Message avec détail des frais
            emoji = "💚" if net_pnl > 0 else "❌"
            message = f"{emoji} <b>POSITION FERMÉE - AVEC FRAIS</b>\n"
            message += f"📊 {product_id}\n"
            message += f"💵 Entrée: ${position['entry_price']:.4f}\n"
            message += f"💵 Sortie: ${current_price:.4f}\n"
            message += f"📈 P&L BRUT: {real_pnl_pct:+.2f}% ({gross_pnl:+.2f}€)\n"
            message += f"💸 Frais entrée: -{entry_fee:.2f}€ (0.5%)\n"
            message += f"💸 Frais sortie: -{exit_fee:.2f}€ (0.5%)\n"
            message += f"💸 Total frais: -{total_fees:.2f}€\n"
            message += f"💰 P&L NET: <b>{net_pnl:+.2f}€</b>\n"
            message += f"🏷️ Raison: {exit_reason}\n"
            message += f"⏱️ Durée: {trade_record['duration']}\n"
            message += f"💰 Capital: {self.current_capital:.2f}€"
            
            # Performance totale
            total_performance = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            message += f"\n📊 Performance totale: {total_performance:+.1f}%"
            message += f"\n💸 Frais totaux payés: {self.total_fees_paid:.2f}€"
            
            self.send_message(message)
            logging.info(f"Position RÉELLE fermée avec frais: {product_id} - P&L NET: {net_pnl:+.2f}€ (frais: {total_fees:.2f}€)")
            
        except Exception as e:
            logging.error(f"Erreur fermeture position avec frais: {e}")

    def reset_daily_counters(self):
        """Reset les compteurs quotidiens si nécessaire"""
        try:
            today = datetime.now().date()
            if today != self.last_trade_reset:
                self.daily_trades_count = 0
                self.daily_pnl = 0.0
                self.last_trade_reset = today
                logging.info("🔄 Compteurs quotidiens réinitialisés")
        except Exception as e:
            logging.error(f"Erreur reset compteurs: {e}")

    def get_portfolio_summary(self):
        """Retourne un résumé du portefeuille"""
        try:
            current_capital = self.initial_capital + self.total_pnl
            performance_pct = ((current_capital - self.initial_capital) / self.initial_capital) * 100
            
            return {
                'initial_capital': self.initial_capital,
                'current_capital': current_capital,
                'total_pnl': self.total_pnl,
                'performance_pct': performance_pct,
                'active_positions': len(self.active_positions),
                'daily_trades': self.daily_trades_count,
                'total_trades': len(self.trade_history),
                'strategy': self.strategy_config['strategy_name']
            }
        except Exception as e:
            logging.error(f"Erreur résumé portefeuille: {e}")
            return {}

    def run_adaptive_analysis(self):
        """Lance l'analyse adaptée à la stratégie - CORRIGÉE"""
        try:
            # Reset compteurs si nouveau jour
            self.reset_daily_counters()
            
            strategy_name = self.strategy_config["strategy_name"]
            logging.info(f"🔄 Analyse {strategy_name} en cours...")
            
            trades_executed = 0
            
            # Analyser les cryptos selon l'allocation
            for product_id, crypto_config in self.specialized_cryptos.items():
                try:
                    # Vérifier si on a atteint la limite quotidienne
                    if self.daily_trades_count >= self.max_daily_trades:
                        logging.info(f"Limite quotidienne atteinte: {self.max_daily_trades} trades")
                        break
                    
                    analysis = self.analyze_adaptive_signal(product_id, crypto_config)
                    if analysis and analysis['signal_type'] in ['BUY', 'SELL']:
                        # Vérifier que le signal est assez fort
                        if analysis['signal_strength'] >= self.min_signal_strength:
                            self.execute_adaptive_trade(analysis, crypto_config)
                            trades_executed += 1
                        else:
                            logging.info(f"Signal {product_id} trop faible: {analysis['signal_strength']}/{self.min_signal_strength}")
                    
                    # Pause adaptée à la stratégie
                    pause_time = 0.5 if strategy_name == "VERY_AGGRESSIVE" else 1
                    time.sleep(pause_time)
                    
                except Exception as e:
                    logging.error(f"Erreur analyse {product_id}: {e}")
                    continue
            
            # Vérifier les conditions de sortie
            self.check_adaptive_exit_conditions()
            
            # Message de fin d'analyse (seulement si des trades exécutés ou positions actives)
            if trades_executed > 0 or len(self.active_positions) > 0:
                current_capital = self.initial_capital + self.total_pnl
                performance_pct = ((current_capital - self.initial_capital) / self.initial_capital) * 100
                
                # Affichage de la fréquence corrigée
                freq_display = f"{self.analysis_frequency}min" if self.analysis_frequency >= 1 else f"{int(self.analysis_frequency*60)}s"
                
                message = f"✅ <b>ANALYSE {strategy_name}</b>\n"
                message += f"🎯 {trades_executed} nouveau(x) trade(s)\n"
                message += f"📊 {len(self.active_positions)} position(s) active(s)\n"
                message += f"💰 Capital: {current_capital:.2f}€ ({performance_pct:+.1f}%)\n"
                message += f"⏰ Prochaine analyse dans {freq_display}"
                
                self.send_message(message)
            
            # Vérifier si on doit adapter la stratégie (capital a changé significativement)
            self.check_strategy_adaptation()
            
        except Exception as e:
            logging.error(f"Erreur analyse adaptative: {e}")

    def analyze_ultra_conservatrice(self, product_id, crypto_config, tf_data):
        """Analyse ultra-conservatrice pour petit capital"""
        try:
            # Tendance générale (1H)
            h1_data = tf_data.get('1h')
            if h1_data is None or len(h1_data) < 50:
                return None
                
            h1_latest = h1_data.iloc[-1]
            
            # Vérifier que les colonnes EMA existent
            required_ema_cols = ['ema_12', 'ema_26', 'ema_50']
            for col in required_ema_cols:
                if col not in h1_data.columns or pd.isna(h1_latest[col]):
                    logging.warning(f"Colonne EMA {col} manquante pour {product_id}")
                    return None
            
            # Tendance forte requise
            bullish_trend = (h1_latest['ema_12'] > h1_latest['ema_26'] > h1_latest['ema_50'])
            bearish_trend = (h1_latest['ema_12'] < h1_latest['ema_26'] < h1_latest['ema_50'])
            
            if not bullish_trend and not bearish_trend:
                return None  # Pas de tendance claire
            
            # Signal d'entrée (15M)
            m15_data = tf_data.get('15m')
            if m15_data is None or len(m15_data) < 30:
                return None
                
            m15_latest = m15_data.iloc[-1]
            m15_prev = m15_data.iloc[-2]
            
            # Vérifier les données requises
            required_cols = ['rsi', 'volume_ratio', 'macd', 'macd_signal']
           
            for col in required_cols:
                if col not in m15_data.columns or pd.isna(m15_latest[col]):
                    logging.warning(f"Colonne {col} manquante ou NaN pour {product_id}")
                    return None
            
            # Critères ultra-stricts
            signals = []
            
            # RSI extrême avec divergence
            if m15_latest['rsi'] < 25 and m15_prev['rsi'] < m15_latest['rsi'] and bullish_trend:
                signals.append({'type': 'BUY', 'strength': 3, 'reason': 'RSI survente extrême + trend bull'})
            elif m15_latest['rsi'] > 75 and m15_prev['rsi'] > m15_latest['rsi'] and bearish_trend:
                signals.append({'type': 'SELL', 'strength': 3, 'reason': 'RSI surachat extrême + trend bear'})
            
            # Volume confirmation obligatoire
            if m15_latest['volume_ratio'] < 2.0:
                return None
            
            signals.append({'type': 'VOLUME', 'strength': 2, 'reason': f'Volume {m15_latest["volume_ratio"]:.1f}x'})
            
            # MACD confirmation
            if (m15_latest['macd'] > m15_latest['macd_signal'] and 
                m15_prev['macd'] <= m15_prev['macd_signal'] and bullish_trend):
                signals.append({'type': 'BUY', 'strength': 2, 'reason': 'MACD bullish cross'})
            elif (m15_latest['macd'] < m15_latest['macd_signal'] and 
                  m15_prev['macd'] >= m15_prev['macd_signal'] and bearish_trend):
                signals.append({'type': 'SELL', 'strength': 2, 'reason': 'MACD bearish cross'})
            
            # Au moins 6 points requis (réduit de 7 pour plus de signaux)
            total_strength = sum(s['strength'] for s in signals)
            if total_strength < 6:
                return None
            
            # Déterminer le type de signal
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            if buy_signals and len(buy_signals) >= len(sell_signals):
                signal_type = 'BUY'
            elif sell_signals:
                signal_type = 'SELL'
            else:
                return None
            
            return {
                'product_id': product_id,
                'signal_type': signal_type,
                'signal_strength': total_strength,
                'current_price': m15_latest['close'],
                'reasons': [s['reason'] for s in signals]
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse ultra-conservatrice: {e}")
            return None

    def analyze_conservative(self, product_id, crypto_config, tf_data):
        """Analyse conservatrice pour capital moyen"""
        try:
            # Tendance 1H + Signal 5M
            h1_data = tf_data.get('1h')
            m5_data = tf_data.get('5m')
            
            if h1_data is None or m5_data is None:
                return None
            
            if len(h1_data) < 30 or len(m5_data) < 50:
                return None
            
            h1_latest = h1_data.iloc[-1]
            m5_latest = m5_data.iloc[-1]
            m5_prev = m5_data.iloc[-2]
            
            signals = []
            
            # Tendance H1 (plus flexible)
            if 'ema_9' in h1_data.columns and 'ema_21' in h1_data.columns:
                if h1_latest['ema_9'] > h1_latest['ema_21']:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Tendance H1 haussière'})
                elif h1_latest['ema_9'] < h1_latest['ema_21']:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Tendance H1 baissière'})
            
            # RSI 5M (seuils plus souples)
            if 'rsi' in m5_data.columns:
                if m5_latest['rsi'] < 35 and m5_prev['rsi'] < m5_latest['rsi']:
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'RSI survente 5M'})
                elif m5_latest['rsi'] > 65 and m5_prev['rsi'] > m5_latest['rsi']:
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'RSI surachat 5M'})
            
            # MACD 5M
            if all(col in m5_data.columns for col in ['macd', 'macd_signal']):
                if (m5_latest['macd'] > m5_latest['macd_signal'] and 
                    m5_prev['macd'] <= m5_prev['macd_signal']):
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'MACD cross bull 5M'})
                elif (m5_latest['macd'] < m5_latest['macd_signal'] and 
                      m5_prev['macd'] >= m5_prev['macd_signal']):
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'MACD cross bear 5M'})
            
            # Volume confirmation (plus souple)
            if 'volume_ratio' in m5_data.columns and m5_latest['volume_ratio'] > 1.5:
                signals.append({'type': 'VOLUME', 'strength': 1, 'reason': f'Volume {m5_latest["volume_ratio"]:.1f}x'})
            
            # Bollinger Bands
            if all(col in m5_data.columns for col in ['bb_upper', 'bb_lower', 'close']):
                if m5_latest['close'] < m5_latest['bb_lower']:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Prix sous BB inf'})
                elif m5_latest['close'] > m5_latest['bb_upper']:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Prix sur BB sup'})
            
            # Au moins 5 points requis
            total_strength = sum(s['strength'] for s in signals)
            if total_strength < 5:
                return None
            
            # Déterminer le signal
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            if len(buy_signals) > len(sell_signals):
                signal_type = 'BUY'
            elif len(sell_signals) > len(buy_signals):
                signal_type = 'SELL'
            else:
                return None
            
            return {
                'product_id': product_id,
                'signal_type': signal_type,
                'signal_strength': total_strength,
                'current_price': m5_latest['close'],
                'reasons': [s['reason'] for s in signals]
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse conservative: {e}")
            return None

    def analyze_conservative_extreme(self, product_id, crypto_config, tf_data):
        """Analyse conservatrice EXTRÊME - OPTIMISÉE CONTRE LES FRAIS"""
        try:
            # Analyse 5M + 1M avec critères renforcés
            m5_data = tf_data.get('5m')
            m1_data = tf_data.get('1m')
            
            if not all([m5_data is not None, m1_data is not None]):
                return None
            
            if any(len(data) < 20 for data in [m5_data, m1_data]):
                return None
            
            m5_latest = m5_data.iloc[-1]
            m1_latest = m1_data.iloc[-1]
            m1_prev = m1_data.iloc[-2]
            m5_prev = m5_data.iloc[-2]
            
            signals = []
            
            # RSI avec momentum (plus strict)
            if 'rsi' in m1_data.columns:
                # Confirmation avec momentum
                if m1_latest['rsi'] < 40 and m1_latest['rsi'] > m1_prev['rsi']:  # Rebond
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'RSI rebond 40-'})
                elif m1_latest['rsi'] > 60 and m1_latest['rsi'] < m1_prev['rsi']:  # Retournement
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'RSI retournement 60+'})
            
            # MACD avec confirmation (plus strict)
            if all(col in m1_data.columns for col in ['macd', 'macd_signal']):
                # Cross MACD uniquement
                if (m1_latest['macd'] > m1_latest['macd_signal'] and 
                    m1_prev['macd'] <= m1_prev['macd_signal']):
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'MACD cross bull'})
                elif (m1_latest['macd'] < m1_latest['macd_signal'] and 
                      m1_prev['macd'] >= m1_prev['macd_signal']):
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'MACD cross bear'})
            
            # EMA avec confirmation 5M (plus strict)
            if all(col in m5_data.columns for col in ['ema_9', 'ema_21']):
                # Cross EMA récent
                if (m5_latest['ema_9'] > m5_latest['ema_21'] and 
                    m5_prev['ema_9'] <= m5_prev['ema_21']):
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'EMA cross bull 5M'})
                elif (m5_latest['ema_9'] < m5_latest['ema_21'] and 
                      m5_prev['ema_9'] >= m5_prev['ema_21']):
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'EMA cross bear 5M'})
            
            # Volume OBLIGATOIRE et plus strict
            if 'volume_ratio' in m1_data.columns:
                if m1_latest['volume_ratio'] > 1.5:  # 1.5x au lieu de 0.8x
                    signals.append({'type': 'VOLUME', 'strength': 1, 'reason': f'Volume {m1_latest["volume_ratio"]:.1f}x'})
                else:
                    # Pas assez de volume = pas de trade
                    return None
            
            # 4 points requis au lieu de 3 (plus selectif)
            total_strength = sum(s['strength'] for s in signals)
            if total_strength < 4:
                return None
            
            # Déterminer le signal
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            # Nécessite au moins 2 signaux du même type
            if len(buy_signals) >= 2 and len(buy_signals) > len(sell_signals):
                signal_type = 'BUY'
            elif len(sell_signals) >= 2 and len(sell_signals) > len(buy_signals):
                signal_type = 'SELL'
            else:
                return None
            
            return {
                'product_id': product_id,
                'signal_type': signal_type,
                'signal_strength': total_strength,
                'current_price': m1_latest['close'],
                'reasons': [s['reason'] for s in signals]
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse conservative extreme: {e}")
            return None

    def analyze_balanced(self, product_id, crypto_config, tf_data):
        """Analyse équilibrée pour capital moyen-élevé"""
        try:
            # Multi-timeframe: 1H + 15M + 5M
            h1_data = tf_data.get('1h')
            m15_data = tf_data.get('15m')
            m5_data = tf_data.get('5m')
            
            if not all([h1_data is not None, m15_data is not None, m5_data is not None]):
                return None
            
            if any(len(data) < 30 for data in [h1_data, m15_data, m5_data]):
                return None
            
            h1_latest = h1_data.iloc[-1]
            m15_latest = m15_data.iloc[-1]
            m5_latest = m5_data.iloc[-1]
            m5_prev = m5_data.iloc[-2]
            
            signals = []
            
            # Tendance H1
            if all(col in h1_data.columns for col in ['ema_9', 'ema_21', 'ema_50']):
                if h1_latest['ema_9'] > h1_latest['ema_21'] > h1_latest['ema_50']:
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'Forte tendance H1 bull'})
                elif h1_latest['ema_9'] < h1_latest['ema_21'] < h1_latest['ema_50']:
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'Forte tendance H1 bear'})
                elif h1_latest['ema_9'] > h1_latest['ema_21']:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Tendance H1 bull'})
                elif h1_latest['ema_9'] < h1_latest['ema_21']:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Tendance H1 bear'})
            
            # RSI multi-timeframe
            if 'rsi' in m15_data.columns and 'rsi' in m5_data.columns:
                if m15_latest['rsi'] < 40 and m5_latest['rsi'] < 40:
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'RSI survente multi-TF'})
                elif m15_latest['rsi'] > 60 and m5_latest['rsi'] > 60:
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'RSI surachat multi-TF'})
            
            # MACD 5M avec confirmation
            if all(col in m5_data.columns for col in ['macd', 'macd_signal']):
                if (m5_latest['macd'] > m5_latest['macd_signal'] and 
                    m5_prev['macd'] <= m5_prev['macd_signal'] and
                    m5_latest['macd'] > 0):
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'MACD bull + momentum'})
                elif (m5_latest['macd'] < m5_latest['macd_signal'] and 
                      m5_prev['macd'] >= m5_prev['macd_signal'] and
                      m5_latest['macd'] < 0):
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'MACD bear + momentum'})
            
            # Volume et volatilité
            if all(col in m5_data.columns for col in ['volume_ratio', 'atr']):
                if m5_latest['volume_ratio'] > 1.5:
                    signals.append({'type': 'VOLUME', 'strength': 1, 'reason': f'Volume {m5_latest["volume_ratio"]:.1f}x'})
            
            # Support/Résistance
            if all(col in m5_data.columns for col in ['support', 'resistance', 'close']):
                if m5_latest['close'] <= m5_latest['support'] * 1.01:  # Near support
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Proche support'})
                elif m5_latest['close'] >= m5_latest['resistance'] * 0.99:  # Near resistance
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Proche résistance'})
            
            # Au moins 4 points requis
            total_strength = sum(s['strength'] for s in signals)
            if total_strength < 4:
                return None
            
            # Déterminer le signal
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            buy_strength = sum(s['strength'] for s in buy_signals)
            sell_strength = sum(s['strength'] for s in sell_signals)
            
            if buy_strength > sell_strength:
                signal_type = 'BUY'
            elif sell_strength > buy_strength:
                signal_type = 'SELL'
            else:
                return None
            
            return {
                'product_id': product_id,
                'signal_type': signal_type,
                'signal_strength': total_strength,
                'current_price': m5_latest['close'],
                'reasons': [s['reason'] for s in signals]
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse balanced: {e}")
            return None

    def analyze_aggressive(self, product_id, crypto_config, tf_data):
        """Analyse agressive pour gros capital"""
        try:
            # Multi-timeframe rapide: 15M + 5M + 1M
            m15_data = tf_data.get('15m')
            m5_data = tf_data.get('5m')
            m1_data = tf_data.get('1m')
            
            if not all([m15_data is not None, m5_data is not None, m1_data is not None]):
                return None
            
            if any(len(data) < 20 for data in [m15_data, m5_data, m1_data]):
                return None
            
            m15_latest = m15_data.iloc[-1]
            m5_latest = m5_data.iloc[-1]
            m1_latest = m1_data.iloc[-1]
            m1_prev = m1_data.iloc[-2]
            
            signals = []
            
            # Tendance rapide (15M)
            if all(col in m15_data.columns for col in ['ema_5', 'ema_13']):
                if m15_latest['ema_5'] > m15_latest['ema_13']:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Tendance 15M bull'})
                else:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Tendance 15M bear'})
            
            # RSI rapide (5M + 1M)
            if 'rsi' in m5_data.columns:
                if m5_latest['rsi'] < 45:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'RSI 5M oversold'})
                elif m5_latest['rsi'] > 55:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'RSI 5M overbought'})
            
            if 'rsi' in m1_data.columns:
                if m1_latest['rsi'] < 30 and m1_prev['rsi'] < m1_latest['rsi']:
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'RSI 1M reversal bull'})
                elif m1_latest['rsi'] > 70 and m1_prev['rsi'] > m1_latest['rsi']:
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'RSI 1M reversal bear'})
            
            # MACD ultra-rapide (1M)
            if all(col in m1_data.columns for col in ['macd', 'macd_signal']):
                if (m1_latest['macd'] > m1_latest['macd_signal'] and 
                    m1_prev['macd'] <= m1_prev['macd_signal']):
                    signals.append({'type': 'BUY', 'strength': 2, 'reason': 'MACD 1M cross bull'})
                elif (m1_latest['macd'] < m1_latest['macd_signal'] and 
                      m1_prev['macd'] >= m1_prev['macd_signal']):
                    signals.append({'type': 'SELL', 'strength': 2, 'reason': 'MACD 1M cross bear'})
            
            # Stochastic ultra-rapide
            if all(col in m1_data.columns for col in ['stoch_k', 'stoch_d']):
                if m1_latest['stoch_k'] < 25:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Stoch < 25'})
                elif m1_latest['stoch_k'] > 75:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Stoch > 75'})
            
            # Williams %R
            if 'williams_r' in m1_data.columns:
                if m1_latest['williams_r'] < -80:
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Williams R oversold'})
                elif m1_latest['williams_r'] > -20:
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Williams R overbought'})
            
            # Volume OBLIGATOIRE pour mode EXTRÊME
            if 'volume_ratio' in m1_data.columns:
                if m1_latest['volume_ratio'] > 1.0:
                    signals.append({'type': 'VOLUME', 'strength': 1, 'reason': f'Volume 1M {m1_latest["volume_ratio"]:.1f}x'})
                else:
                    # Pas assez de volume = pas de trade en mode EXTRÊME
                    return None
            
            # Momentum ultra-sensible
            if 'momentum' in m1_data.columns:
                if m1_latest['momentum'] > 0.001:  # +0.1%
                    signals.append({'type': 'BUY', 'strength': 1, 'reason': 'Micro momentum +'})
                elif m1_latest['momentum'] < -0.001:  # -0.1%
                    signals.append({'type': 'SELL', 'strength': 1, 'reason': 'Micro momentum -'})
            
            # Seulement 2 points requis (TRÈS agressif)
            total_strength = sum(s['strength'] for s in signals)
            if total_strength < 2:
                return None
            
            # Déterminer le signal (plus permissif)
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            buy_strength = sum(s['strength'] for s in buy_signals)
            sell_strength = sum(s['strength'] for s in sell_signals)
            
            if buy_strength > sell_strength:
                signal_type = 'BUY'
            elif sell_strength > buy_strength:
                signal_type = 'SELL'
            else:
                # En mode EXTRÊME, on peut prendre des signaux même équilibrés
                if len(buy_signals) > len(sell_signals):
                    signal_type = 'BUY'
                elif len(sell_signals) > len(buy_signals):
                    signal_type = 'SELL'
                else:
                    return None
            
            return {
                'product_id': product_id,
                'signal_type': signal_type,
                'signal_strength': total_strength,
                'current_price': m1_latest['close'],
                'reasons': [s['reason'] for s in signals]
            }
            
        except Exception as e:
            logging.error(f"Erreur analyse very aggressive: {e}")
            return None

if __name__ == "__main__":
    """Point d'entrée principal du bot adaptatif"""
    try:
        print("🚀 Démarrage du Bot Adaptatif de Trading...")
        
        # Demander le capital initial
        while True:
            try:
                capital_input = input("💰 Entrez votre capital initial (€): ")
                initial_capital = float(capital_input)
                
                if initial_capital <= 0:
                    print("❌ Le capital doit être positif!")
                    continue
                    
                break
            except ValueError:
                print("❌ Veuillez entrer un nombre valide!")
        
        print(f"✅ Capital configuré: {initial_capital:.2f}€")
        
        # Vérifier les variables d'environnement
        load_dotenv()
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not telegram_token:
            print("❌ TELEGRAM_BOT_TOKEN manquant dans le fichier .env")
            print("📝 Créez un fichier .env avec:")
            print("TELEGRAM_BOT_TOKEN=votre_token_ici")
            print("TELEGRAM_CHAT_ID=votre_chat_id_ici")
            exit(1)
            
        if not chat_id:
            print("❌ TELEGRAM_CHAT_ID manquant dans le fichier .env")
            exit(1)
        
        print("✅ Configuration Telegram trouvée")
        
        # Initialiser et démarrer le bot
        bot = AdaptiveTradingBot(initial_capital)
        
        # Afficher la stratégie sélectionnée
        strategy_info = bot.strategy_config
        print(f"\n🎯 Stratégie sélectionnée: {strategy_info['strategy_name']}")
        print(f"📊 Niveau de risque: {strategy_info['risk_level']}")
        print(f"💸 Risque par trade: {strategy_info['risk_per_trade']*100:.1f}%")
        print(f"🔢 Trades max/jour: {strategy_info['max_daily_trades']}")
        print(f"📈 Cryptos: {len(strategy_info['cryptos'])}")
        
        # Avertissement selon le capital
        if initial_capital <= 100:
            print("\n⚠️  MODE ULTRA-CONSERVATEUR AGRESSIF")
            print("🔥 Risque doublé pour maximiser les gains!")
        elif initial_capital <= 500:
            print("\n💀 MODE CONSERVATEUR EXTRÊME 25%")
            print("🔥 RISQUE MAXIMUM: 25% par trade!")
            print("⚠️ ATTENTION: Risque de perte totale élevé!")
        elif initial_capital <= 1000:
            print("\n⚠️  MODE ÉQUILIBRÉ TRÈS AGRESSIF")
            print("🔥 Risque doublé pour performance maximale!")
        elif initial_capital <= 5000:
            print("\n⚠️  MODE AGRESSIF EXTRÊME")
            print("🔥 Risque triplé - Gains potentiels énormes!")
        else:
            print("\n💀 MODE EXTRÊME MAINTENU")
            print("🔥 RISQUE MAXIMUM - Gains ou pertes explosives!")
        
        print(f"\n📊 Timeframes utilisés: {list(bot.timeframes.keys())}")
        
        # Demander confirmation
        print(f"\n🤖 Le bot va démarrer avec cette configuration.")
        confirm = input("▶️  Appuyez sur ENTRÉE pour démarrer (ou 'q' pour quitter): ")
        
        if confirm.lower() == 'q':
            print("👋 Arrêt du programme")
            exit(0)
        
        print("\n🚀 Démarrage du bot adaptatif...")
        print("📱 Vérifiez Telegram pour les notifications")
        print("⚠️  Appuyez sur CTRL+C pour arrêter le bot")
        
        # Démarrer le bot
        bot.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du bot demandé par l'utilisateur")
        try:
            final_capital = bot.initial_capital + bot.total_pnl
            final_performance = ((final_capital - bot.initial_capital) / bot.initial_capital) * 100
            print(f"💰 Capital final: {final_capital:.2f}€")
            print(f"📊 Performance: {final_performance:+.1f}%")
            print(f"🔢 Trades effectués: {len(bot.trade_history)}")
        except:
            pass
        print("👋 Bot arrêté proprement")
        
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        logging.error(f"Erreur critique au démarrage: {e}")
        print("📝 Vérifiez les logs pour plus de détails")
        exit(1)
