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

# Configuration du logging (compatible Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class CoinbaseTelegramBot:
    def __init__(self):
        load_dotenv()
        
        # Configuration Telegram
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.telegram_token or not self.chat_id:
            logging.error("TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquants dans .env")
            raise ValueError("Configuration Telegram manquante")
        
        # Configuration du bot avec nettoyage des commentaires
        self.trading_mode = os.getenv("TRADING_MODE", "SIMULATION")
        
        # Nettoyage des valeurs pour éviter les commentaires
        prediction_threshold_str = os.getenv("PREDICTION_THRESHOLD", "0.00002")
        self.prediction_threshold = float(prediction_threshold_str.split('#')[0].strip())
        
        volume_growth_factor_str = os.getenv("VOLUME_GROWTH_FACTOR", "2")
        self.volume_growth_factor = float(volume_growth_factor_str.split('#')[0].strip())
        
        price_surge_percentage_str = os.getenv("PRICE_SURGE_PERCENTAGE", "0.01")
        self.price_surge_percentage = float(price_surge_percentage_str.split('#')[0].strip())
        
        # Instruments à analyser - LISTE ÉTENDUE
        self.instruments = [
            # TOP CRYPTOS - Tier 1
            "BTC-USD", "ETH-USD", 
            
            # ALTCOINS MAJEURS - Tier 2
            "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD", "XRP-USD", 
            "BCH-USD", "ATOM-USD", "ALGO-USD",
            
            # DEFI & SMART CONTRACTS - Tier 3
            "UNI-USD", "AAVE-USD", "COMP-USD", "MKR-USD", "SNX-USD",
            "CRV-USD", "1INCH-USD", "SUSHI-USD", "YFI-USD",
            
            # LAYER 1 BLOCKCHAINS - Tier 4
            "SOL-USD", "AVAX-USD", "MATIC-USD", "FTM-USD", "NEAR-USD",
            "LUNA-USD", "EGLD-USD", "HBAR-USD", "ICP-USD",
            
            # GAMING & NFT - Tier 5
            "MANA-USD", "SAND-USD", "AXS-USD", "ENJ-USD", "CHZ-USD",
            
            # INFRASTRUCTURE & ORACLES - Tier 6
            "GRT-USD", "FIL-USD", "AR-USD", "STORJ-USD",
            
            # MEME COINS & COMMUNITY - Tier 7
            "DOGE-USD", "SHIB-USD",
            
            # STABLECOINS & WRAPPED - Tier 8
            "USDC-USD", "USDT-USD", "DAI-USD", "WBTC-USD",
            
            # EMERGING ALTCOINS - Tier 9
            "XTZ-USD", "ZEC-USD", "DASH-USD", "ETC-USD", "XLM-USD",
            "EOS-USD", "NEO-USD", "VET-USD", "IOTA-USD", "ONT-USD",
            
            # PRIVACY COINS - Tier 10
            "ZEC-USD", "DASH-USD",
            
            # ENTERPRISE & UTILITY - Tier 11
            "BAT-USD", "ZRX-USD", "REP-USD", "NMR-USD", "LRC-USD",
            "OMG-USD", "SKL-USD", "CVC-USD", "DNT-USD", "MANA-USD",
            
            # ADDITIONAL OPPORTUNITIES - Tier 12
            "BAND-USD", "REN-USD", "KNC-USD", "BNT-USD", "ANKR-USD",
            "NKN-USD", "OXT-USD", "CGLD-USD", "NU-USD", "CTX-USD"
        ]
        
        # Supprimer les doublons
        self.instruments = list(set(self.instruments))
        
        # Log du nombre total d'instruments
        logging.info(f"📊 {len(self.instruments)} instruments configurés pour l'analyse")
        
        # Stockage des données
        self.models = {}
        self.historical_data = {}
        self.realtime_data = {}
        self.last_alerts = {}
        self.ws = None
        
        # NOUVEAU: Système de suivi des trades
        self.active_positions = {}  # Positions ouvertes
        self.trade_history = []     # Historique des trades
        self.total_profit = 0.0     # Profit total cumulé
        self.win_trades = 0         # Nombre de trades gagnants
        self.loss_trades = 0        # Nombre de trades perdants
        
        # Initialisation du bot Telegram
        try:
            self.bot = Bot(token=self.telegram_token)
            logging.info("✅ Bot Telegram initialisé avec succès")
            logging.info(f"📢 Canal configuré: {self.chat_id}")
        except Exception as e:
            logging.error(f"Erreur initialisation bot Telegram: {e}")
            raise

    async def send_message_async(self, message: str):
        """Envoie un message via Telegram (version asynchrone)"""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logging.info(f"Message Telegram envoyé: {message[:50]}...")
        except TelegramError as e:
            logging.error(f"Erreur Telegram: {e}")
        except Exception as e:
            logging.error(f"Erreur inattendue envoi Telegram: {e}")

    def send_message(self, message: str):
        """Envoie un message via Telegram (version synchrone)"""
        try:
            # Utilisation de requests pour éviter les problèmes d'asyncio
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                logging.info(f"Message Telegram envoyé: {message[:50]}...")
            else:
                logging.error(f"Erreur envoi Telegram: {response.text}")
                
        except Exception as e:
            logging.error(f"Erreur envoi Telegram: {e}")

    def get_coinbase_historical_data(self, product_id, granularity=900, periods=200):
        """Récupère les données historiques de Coinbase Pro"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=granularity * periods)
            
            url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': granularity
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return None
                
            # Conversion en DataFrame
            df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time').reset_index(drop=True)
            
            # Conversion des types
            for col in ['low', 'high', 'open', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df
            
        except Exception as e:
            logging.error(f"Erreur récupération données Coinbase pour {product_id}: {e}")
            return None

    def create_features(self, df):
        """Crée les caractéristiques techniques"""
        if df.empty or len(df) < 50:
            return pd.DataFrame()
        
        try:
            # Indicateurs techniques
            df['returns'] = df['close'].pct_change()
            df['ma_10'] = df['close'].rolling(window=10).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_change'] = df['close'].pct_change(5)  # Changement sur 5 périodes
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
            
            # Bandes de Bollinger
            df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Nettoyage
            df = df.dropna()
            return df
            
        except Exception as e:
            logging.error(f"Erreur création caractéristiques: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, prices, period=14):
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcule le MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcule les Bandes de Bollinger"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower

    def train_model(self, df, product_id):
        """Entraîne le modèle de prédiction"""
        if df.empty or len(df) < 100:
            return None
        
        try:
            features = ['ma_10', 'ma_20', 'ma_50', 'volatility', 'rsi', 'volume_ratio', 
                       'macd', 'macd_signal', 'macd_hist', 'bb_position', 'price_change']
            
            # Vérification que toutes les caractéristiques existent
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logging.warning(f"Caractéristiques manquantes pour {product_id}: {missing_features}")
                return None
            
            X = df[features].values
            y = df['returns'].shift(-1).dropna().values  # Prédire le prochain retour
            
            # Ajuster X pour correspondre à y
            X = X[:-1]
            
            if len(X) < 50:
                return None
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Entraînement
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Évaluation
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logging.info(f"Modèle entraîné pour {product_id} - MSE: {mse:.6f}")
            
            return model
            
        except Exception as e:
            logging.error(f"Erreur entraînement modèle pour {product_id}: {e}")
            return None

    def analyze_instrument(self, product_id):
        """Analyse un instrument et génère des signaux"""
        try:
            logging.info(f"📊 Analyse de {product_id}...")
            
            # Récupération des données historiques
            df = self.get_coinbase_historical_data(product_id)
            if df is None or df.empty:
                logging.warning(f"Aucune donnée pour {product_id}")
                return
            
            # Création des caractéristiques
            df_features = self.create_features(df)
            if df_features.empty:
                logging.warning(f"Impossible de créer les caractéristiques pour {product_id}")
                return
            
            # Entraînement du modèle
            model = self.train_model(df_features, product_id)
            if model is None:
                logging.warning(f"Impossible d'entraîner le modèle pour {product_id}")
                return
            
            # Stockage pour utilisation temps réel
            self.models[product_id] = model
            self.historical_data[product_id] = df_features
            
            # Prédiction sur les dernières données
            self.make_prediction(product_id, df_features.iloc[-1])
            
        except Exception as e:
            logging.error(f"Erreur analyse {product_id}: {e}")

    def make_prediction(self, product_id, latest_data):
        """Fait une prédiction et génère des alertes"""
        try:
            if product_id not in self.models:
                return
            
            model = self.models[product_id]
            features = ['ma_10', 'ma_20', 'ma_50', 'volatility', 'rsi', 'volume_ratio', 
                       'macd', 'macd_signal', 'macd_hist', 'bb_position', 'price_change']
            
            # Préparation des données pour prédiction
            X = latest_data[features].values.reshape(1, -1)
            prediction = model.predict(X)[0]
            
            # Récupération du prix actuel
            current_price = latest_data['close']
            current_volume = latest_data['volume']
            
            # Détection des signaux
            signals = []
            
            # Signal de prédiction avec seuil de confiance à 50%
            if abs(prediction) > self.prediction_threshold:
                confidence = min(abs(prediction) * 10000, 100)  # Score de confiance
                
                # FILTRE: Uniquement les signaux > 50% de confiance
                if confidence > 50:
                    direction = "BUY" if prediction > 0 else "SELL"
                    signals.append({
                        'type': 'PREDICTION',
                        'direction': direction,
                        'prediction': prediction,
                        'confidence': confidence
                    })
            
            # Signal de volume (toujours affiché car important)
            volume_ma = latest_data['volume_ma']
            if current_volume > volume_ma * self.volume_growth_factor:
                signals.append({
                    'type': 'VOLUME',
                    'direction': 'HIGH_VOLUME',
                    'ratio': current_volume / volume_ma
                })
            
            # Signal RSI avec recommandations
            rsi = latest_data['rsi']
            if rsi < 30:
                signals.append({
                    'type': 'RSI', 
                    'direction': 'OVERSOLD', 
                    'value': rsi,
                    'action': 'BUY_OPPORTUNITY'
                })
            elif rsi > 70:
                signals.append({
                    'type': 'RSI', 
                    'direction': 'OVERBOUGHT', 
                    'value': rsi,
                    'action': 'SELL_OPPORTUNITY'
                })
            
            # Signal Bollinger Bands avec recommandations
            bb_position = latest_data['bb_position']
            if bb_position < 0.1:
                signals.append({
                    'type': 'BB', 
                    'direction': 'NEAR_LOWER_BAND', 
                    'value': bb_position,
                    'action': 'BUY_OPPORTUNITY'
                })
            elif bb_position > 0.9:
                signals.append({
                    'type': 'BB', 
                    'direction': 'NEAR_UPPER_BAND', 
                    'value': bb_position,
                    'action': 'SELL_OPPORTUNITY'
                })
            
            # MACD Signal
            macd = latest_data['macd']
            macd_signal = latest_data['macd_signal']
            if macd > macd_signal and macd > 0:
                signals.append({
                    'type': 'MACD',
                    'direction': 'BULLISH_CROSSOVER',
                    'action': 'BUY_SIGNAL'
                })
            elif macd < macd_signal and macd < 0:
                signals.append({
                    'type': 'MACD',
                    'direction': 'BEARISH_CROSSOVER',
                    'action': 'SELL_SIGNAL'
                })
            
            # Envoi des alertes
            if signals:
                self.send_detailed_trading_alert(product_id, current_price, latest_data, signals)
                
        except Exception as e:
            logging.error(f"Erreur prédiction {product_id}: {e}")

    def track_buy_signal(self, product_id, price, signals):
        """Enregistre un signal d'achat"""
        try:
            position = {
                'product_id': product_id,
                'action': 'BUY',
                'entry_price': price,
                'entry_time': datetime.now(),
                'quantity': 1.0,  # Simulation: 1 unité
                'signals': [s['type'] for s in signals],
                'confidence': max([s.get('confidence', 0) for s in signals if 'confidence' in s], default=0)
            }
            
            self.active_positions[product_id] = position
            logging.info(f"📈 Position ACHAT ouverte: {product_id} à ${price:.4f}")
            
        except Exception as e:
            logging.error(f"Erreur suivi achat {product_id}: {e}")

    def track_sell_signal(self, product_id, current_price):
        """Enregistre un signal de vente et calcule le profit"""
        try:
            if product_id not in self.active_positions:
                # Nouvelle position de vente (short)
                position = {
                    'product_id': product_id,
                    'action': 'SELL',
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'quantity': 1.0,
                    'signals': ['SELL'],
                    'confidence': 0
                }
                self.active_positions[product_id] = position
                logging.info(f"📉 Position VENTE ouverte: {product_id} à ${current_price:.4f}")
                return None
                
            # Fermeture d'une position d'achat existante
            position = self.active_positions[product_id]
            
            if position['action'] == 'BUY':
                # Calcul du profit sur position longue
                profit = (current_price - position['entry_price']) * position['quantity']
                profit_percentage = ((current_price - position['entry_price']) / position['entry_price']) * 100
                
                # Enregistrement du trade
                trade = {
                    'product_id': product_id,
                    'action': 'BUY_TO_SELL',
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'quantity': position['quantity'],
                    'profit_usd': profit,
                    'profit_pct': profit_percentage,
                    'duration': datetime.now() - position['entry_time'],
                    'entry_signals': position['signals'],
                    'confidence': position['confidence']
                }
                
                self.trade_history.append(trade)
                self.total_profit += profit
                
                if profit > 0:
                    self.win_trades += 1
                else:
                    self.loss_trades += 1
                
                # Supprimer la position active
                del self.active_positions[product_id]
                
                logging.info(f"💰 Trade fermé: {product_id} - Profit: ${profit:.2f} ({profit_percentage:.2f}%)")
                return trade
                
        except Exception as e:
            logging.error(f"Erreur suivi vente {product_id}: {e}")
            return None

    def get_performance_stats(self):
        """Calcule les statistiques de performance"""
        try:
            total_trades = len(self.trade_history)
            if total_trades == 0:
                return None
                
            win_rate = (self.win_trades / total_trades) * 100
            
            profits = [t['profit_usd'] for t in self.trade_history if t['profit_usd'] > 0]
            losses = [t['profit_usd'] for t in self.trade_history if t['profit_usd'] < 0]
            
            avg_win = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            best_trade = max(self.trade_history, key=lambda x: x['profit_usd']) if self.trade_history else None
            worst_trade = min(self.trade_history, key=lambda x: x['profit_usd']) if self.trade_history else None
            
            return {
                'total_trades': total_trades,
                'win_trades': self.win_trades,
                'loss_trades': self.loss_trades,
                'win_rate': win_rate,
                'total_profit': self.total_profit,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'active_positions': len(self.active_positions)
            }
            
        except Exception as e:
            logging.error(f"Erreur calcul stats: {e}")
            return None

    def send_detailed_trading_alert(self, product_id, price, data, signals):
        """Envoie une alerte de trading détaillée via Telegram"""
        try:
            # Éviter le spam - MODIFIÉ: Réduire à 2 minutes pour débugger
            now = time.time()
            if product_id in self.last_alerts:
                if now - self.last_alerts[product_id] < 120:  # 2 minutes au lieu de 5
                    logging.info(f"⏰ Alerte {product_id} bloquée par anti-spam (dernière: {(now - self.last_alerts[product_id])/60:.1f}min)")
                    return
            
            self.last_alerts[product_id] = now
            
            # DEBUG: Log pour vérifier qu'on arrive ici
            logging.info(f"🔔 Préparation alerte pour {product_id} - Prix: ${price:.4f}")
            
            # Analyser les signaux pour déterminer l'action principale
            buy_signals = 0
            sell_signals = 0
            main_action = "⏳ ATTENDRE"
            action_strength = 0
            
            for signal in signals:
                if signal.get('action') in ['BUY_OPPORTUNITY', 'BUY_SIGNAL'] or signal.get('direction') == 'BUY':
                    buy_signals += 1
                elif signal.get('action') in ['SELL_OPPORTUNITY', 'SELL_SIGNAL'] or signal.get('direction') == 'SELL':
                    sell_signals += 1
            
            # Déterminer l'action principale avec priorité RSI extrême
            rsi_value = data['rsi']
            if rsi_value > 75:
                main_action = "🔴 VENDRE"
                action_strength = 1
            elif rsi_value < 25:
                main_action = "🟢 ACHETER"
                action_strength = 1
            elif buy_signals > sell_signals:
                main_action = "🟢 ACHETER"
                action_strength = buy_signals
            elif sell_signals > buy_signals:
                main_action = "🔴 VENDRE"
                action_strength = sell_signals

            # DEBUG: Log pour voir l'action déterminée
            logging.info(f"🎯 Action déterminée pour {product_id}: {main_action} (Force: {action_strength})")
            logging.info(f"📊 Signaux - Achat: {buy_signals}, Vente: {sell_signals}, RSI: {rsi_value:.1f}")

            # FILTRE: N'envoyer que les alertes d'ACHAT
            if main_action != "🟢 ACHETER":
                logging.info(f"Signal {main_action} ignoré pour {product_id} - Seuls les achats sont notifiés")
                return

            # DEBUG: Log pour confirmer qu'on va envoyer l'alerte
            logging.info(f"✅ Envoi alerte d'ACHAT pour {product_id}")

            # NOUVEAU: Gérer le suivi des trades
            trade_closed = None
            if main_action == "🟢 ACHETER":
                self.track_buy_signal(product_id, price, signals)
            
            # NOUVEAU: Générer le lien Coinbase
            crypto_symbol = product_id.replace('-USD', '').lower()
            coinbase_link = f"https://www.coinbase.com/price/{crypto_symbol}"
            
            # Construction du message détaillé
            message = f"🚨 <b>SIGNAL DE TRADING</b> 🚨\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"💰 <b>{product_id}</b>\n"
            message += f"💵 Prix actuel: <b>${price:.4f}</b>\n"
            message += f"📊 <a href='{coinbase_link}'>Voir le graphique Coinbase</a>\n"
            message += f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            
            # NOUVEAU: Calcul et affichage des frais Crypto.com
            message += f"\n💳 <b>FRAIS CRYPTO.COM (0.4% par transaction):</b>\n"
            
            # Calcul pour différents montants
            amounts = [100, 500, 1000]
            for amount in amounts:
                fees_data = self.calculate_crypto_com_fees(amount)
                if fees_data:
                    message += f"💶 <b>{amount}€:</b>\n"
                    message += f"   📈 Frais achat: {fees_data['buy_fee']:.2f}€\n"
                    message += f"   📉 Frais vente: {fees_data['sell_fee']:.2f}€\n"
                    message += f"   💸 Total frais: <b>{fees_data['total_fees']:.2f}€</b>\n"
                    message += f"   💰 Net après achat: {fees_data['net_after_buy']:.2f}€\n"
                    message += f"   🔄 Net après A/R: {fees_data['net_after_roundtrip']:.2f}€\n\n"
        
            # Reste du message (simplifié pour débugger)
            message += f"🎯 <b>RECOMMANDATION: {main_action}</b>\n"
            if action_strength > 0:
                message += f"💪 Force du signal: {action_strength} indicateur(s)\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            # Informations de base
            message += f"📋 <b>CONTEXTE MARCHÉ:</b>\n"
            message += f"📊 RSI: {data['rsi']:.1f}\n"
            message += f"💹 Volume: {data['volume_ratio']:.2f}x moyenne\n"
            message += f"📈 MA20: ${data['ma_20']:.4f}\n"
            message += f"📊 Volatilité: {data['volatility']:.4f}\n\n"
            
            message += f"🤖 Mode: {self.trading_mode}"
            message += f"\n⚠️ Ceci n'est pas un conseil financier"
            
            # DEBUG: Log avant envoi
            logging.info(f"📤 Envoi du message Telegram pour {product_id}")
            
            self.send_message(message)
            logging.info(f"✅ Alerte d'ACHAT envoyée pour {product_id}: {len(signals)} signaux - Action: {main_action}")
            
        except Exception as e:
            logging.error(f"❌ Erreur envoi alerte détaillée {product_id}: {e}")
            # DEBUG: Afficher la stack trace complète
            import traceback
            logging.error(f"Stack trace: {traceback.format_exc()}")

    def send_daily_performance_report(self):
        """Envoie un rapport de performance quotidien"""
        try:
            stats = self.get_performance_stats()
            if not stats:
                return
                
            message = f"📊 <b>RAPPORT QUOTIDIEN</b> 📊\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📅 {datetime.now().strftime('%d/%m/%Y')}\n\n"
            
            profit_emoji = "💚" if stats['total_profit'] > 0 else "❌" if stats['total_profit'] < 0 else "⚖️"
            message += f"{profit_emoji} <b>Profit Total: ${stats['total_profit']:.2f}</b>\n\n"
            
            message += f"📋 <b>STATISTIQUES:</b>\n"
            message += f"🔢 Total trades: {stats['total_trades']}\n"
            message += f"✅ Trades gagnants: {stats['win_trades']}\n"
            message += f"❌ Trades perdants: {stats['loss_trades']}\n"
            message += f"🎯 Taux de réussite: <b>{stats['win_rate']:.1f}%</b>\n\n"
            
            if stats['avg_win'] > 0 and stats['avg_loss'] < 0:
                message += f"📈 Gain moyen: ${stats['avg_win']:.2f}\n"
                message += f"📉 Perte moyenne: ${stats['avg_loss']:.2f}\n"
                message += f"⚖️ Ratio Risk/Reward: {abs(stats['avg_win']/stats['avg_loss']):.2f}\n\n"
            
            if stats['best_trade']:
                message += f"🥇 <b>MEILLEUR TRADE:</b>\n"
                message += f"💰 {stats['best_trade']['product_id']}: ${stats['best_trade']['profit_usd']:.2f} ({stats['best_trade']['profit_pct']:.2f}%)\n\n"
            
            if stats['worst_trade']:
                message += f"🥶 <b>PIRE TRADE:</b>\n"
                message += f"💸 {stats['worst_trade']['product_id']}: ${stats['worst_trade']['profit_usd']:.2f} ({stats['worst_trade']['profit_pct']:.2f}%)\n\n"
            
            message += f"📋 Positions actives: {stats['active_positions']}\n"
            message += f"🤖 Mode: {self.trading_mode}"
            
            self.send_message(message)
            logging.info("Rapport de performance quotidien envoyé")
            
        except Exception as e:
            logging.error(f"Erreur rapport performance: {e}")

    def run_analysis_cycle(self):
        """Lance un cycle d'analyse complet"""
        logging.info("🔄 Début du cycle d'analyse...")
        self.send_message("🔄 <b>ANALYSE EN COURS...</b>\n📊 Recherche d'opportunités de trading...")
        
        opportunities = 0
        alerts_sent = 0
        
        for instrument in self.instruments:
            try:
                initial_alerts = len(self.last_alerts)
                self.analyze_instrument(instrument)
                
                # Compter les nouvelles alertes
                if len(self.last_alerts) > initial_alerts:
                    alerts_sent += 1
                    
                opportunities += 1
                time.sleep(2)  # Pause pour éviter les limites de taux
            except Exception as e:
                logging.error(f"Erreur analyse {instrument}: {e}")
        
        # Message de fin d'analyse
        if alerts_sent > 0:
            message = f"✅ <b>ANALYSE TERMINÉE</b>\n"
            message += f"🎯 {alerts_sent} alerte(s) générée(s)\n"
            message += f"📊 {opportunities} instruments analysés\n"
            message += f"🤖 Mode: {self.trading_mode}"
        else:
            message = f"✅ <b>ANALYSE TERMINÉE</b>\n"
            message += f"😴 Aucune opportunité détectée\n"
            message += f"📊 {opportunities} instruments analysés\n"
            message += f"🤖 Mode: {self.trading_mode}"
            
        self.send_message(message)
        logging.info(f"Cycle d'analyse terminé - {opportunities} instruments - {alerts_sent} alertes")

    def send_status_message(self):
        """Envoie un message de statut"""
        active_instruments = len(self.models)
        
        message = f"🤖 <b>BOT STATUS</b>\n"
        message += f"📊 Instruments actifs: {active_instruments}\n"
        message += f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        message += f"🔄 Mode: {self.trading_mode}\n"
        message += f"✅ Bot opérationnel"
        
        self.send_message(message)

    def run(self):
        """Démarre le bot"""
        try:
            # Message de démarrage
            self.send_message("🚀 <b>BOT DE TRADING DÉMARRÉ</b>\n📊 Initialisation en cours...")
            
            # Analyse initiale
            self.run_analysis_cycle()
            
            # Programmation des analyses périodiques
            schedule.every().hour.do(self.run_analysis_cycle)
            schedule.every(4).hours.do(self.send_status_message)
            schedule.every().day.at("08:00").do(self.send_daily_performance_report)  # NOUVEAU: Rapport quotidien
            schedule.every().day.at("09:00").do(self.send_fees_summary)  # Résumé quotidien des frais
            
            # Message de confirmation
            self.send_message("✅ <b>BOT OPÉRATIONNEL</b>\n📊 Analyses programmées toutes les heures\n📱 Vous recevrez les alertes dans ce canal\n💰 Suivi des profits activé")
            
            # Boucle principale
            logging.info("🎯 Bot opérationnel - En attente des signaux...")
            while True:
                schedule.run_pending()
                time.sleep(60)  # Vérification chaque minute
                
        except KeyboardInterrupt:
            logging.info("Arrêt du bot demandé par l'utilisateur")
            final_stats = self.get_performance_stats()
            if final_stats:
                self.send_message(f"🛑 <b>BOT ARRÊTÉ</b>\n💰 Profit Final: ${final_stats['total_profit']:.2f}\n📊 {final_stats['total_trades']} trades - {final_stats['win_rate']:.1f}% de réussite")
        except Exception as e:
            logging.error(f"Erreur critique: {e}")
            self.send_message(f"❌ Erreur critique: {e}")

    def calculate_crypto_com_fees(self, amount_eur):
        """
        Calcule les frais de transaction Crypto.com pour différents montants
        Frais Crypto.com : 0.4% par transaction (maker/taker)
        """
        try:
            # Frais de base Crypto.com (0.4% par transaction)
            base_fee_rate = 0.004
            
            # Frais pour achat
            buy_fee = amount_eur * base_fee_rate
            
            # Frais pour vente (même taux)
            sell_fee = amount_eur * base_fee_rate
            
            # Frais total (achat + vente)
            total_fees = buy_fee + sell_fee
            
            # Montant net après frais d'achat
            net_amount_after_buy = amount_eur - buy_fee
            
            # Montant net après vente complète (roundtrip)
            net_amount_after_roundtrip = amount_eur - total_fees
            
            return {
                'amount_eur': amount_eur,
                'buy_fee': buy_fee,
                'sell_fee': sell_fee,
                'total_fees': total_fees,
                'net_after_buy': net_amount_after_buy,
                'net_after_roundtrip': net_amount_after_roundtrip,
                'fee_percentage': base_fee_rate * 100
            }
            
        except Exception as e:
            logging.error(f"Erreur calcul frais Crypto.com: {e}")
            return None

    def send_fees_summary(self):
        """Envoie un résumé des frais Crypto.com"""
        try:
            message = f"💳 <b>FRAIS CRYPTO.COM - RÉSUMÉ</b>\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            message += f"📊 <b>TAUX DE FRAIS:</b>\n"
            message += f"• Maker: <b>0.40%</b>\n"
            message += f"• Taker: <b>0.40%</b>\n"
            message += f"• Retrait: Variable selon crypto\n\n"
            
            message += f"💰 <b>CALCUL DES FRAIS:</b>\n"
            
            amounts = [100, 500, 1000, 2000, 5000]
            for amount in amounts:
                fees_data = self.calculate_crypto_com_fees(amount)
                if fees_data:
                    message += f"💶 <b>{amount}€:</b>\n"
                    message += f"   📈 Achat: {fees_data['buy_fee']:.2f}€\n"
                    message += f"   📉 Vente: {fees_data['sell_fee']:.2f}€\n"
                    message += f"   🔄 Total A/R: <b>{fees_data['total_fees']:.2f}€</b>\n\n"
            
            message += f"💡 <b>CONSEILS:</b>\n"
            message += f"• Minimum 2% de gain pour couvrir les frais\n"
            message += f"• Éviter les trades très courts\n"
            message += f"• Considérer les frais dans vos calculs\n"
            message += f"• Utiliser les ordres maker si possible\n"
            
            self.send_message(message)
            
        except Exception as e:
            logging.error(f"Erreur envoi résumé frais: {e}")

def main():
    try:
        bot = CoinbaseTelegramBot()
        bot.run()
    except Exception as e:
        logging.error(f"Impossible de démarrer le bot: {e}")
        print(f"❌ ERREUR: {e}")

if __name__ == "__main__":
    main()
