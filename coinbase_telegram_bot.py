import asyncio
import logging
import os
import json
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import websocket
import threading
from dotenv import load_dotenv
import schedule
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging (compatible Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Chargement des variables d'environnement
load_dotenv()

class CoinbaseRealtimeBot:
    def __init__(self):
        # Configuration Telegram
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.telegram_token or not self.chat_id:
            logging.error("TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquants dans .env")
            print("❌ ERREUR: Configuration Telegram manquante")
            print("📝 Vérifiez votre fichier .env :")
            print("   - TELEGRAM_BOT_TOKEN=votre_token_du_botfather") 
            print("   - TELEGRAM_CHAT_ID=votre_chat_id")
            print("💡 Utilisez le script get_chat_id.py pour obtenir votre Chat ID")
            exit("Erreur: Configuration Telegram manquante")
        
        # Test de validation du token
        if self.telegram_token == "your_telegram_bot_token_from_botfather":
            logging.error("Token Telegram par défaut détecté")
            print("❌ ERREUR: Token Telegram non configuré")
            print("🔑 Remplacez 'your_telegram_bot_token_from_botfather' par votre vrai token")
            print("💡 Obtenez votre token via @BotFather sur Telegram")
            exit("Erreur: Token Telegram invalide")
        
        try:
            self.telegram_bot = Bot(token=self.telegram_token)
        except Exception as e:
            logging.error(f"Erreur initialisation bot Telegram: {e}")
            print("❌ ERREUR: Token Telegram invalide ou problème réseau")
            print("🔍 Vérifiez que votre token est correct")
            exit("Erreur: Impossible d'initialiser le bot Telegram")
        
        # Configuration Coinbase
        self.coinbase_api_key = os.getenv("COINBASE_API_KEY")
        self.coinbase_secret = os.getenv("COINBASE_API_SECRET")
        self.coinbase_passphrase = os.getenv("COINBASE_PASSPHRASE")
        
        # Paramètres du bot
        self.trading_mode = os.getenv("TRADING_MODE", "SIMULATION")
        self.prediction_threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.00002"))
        self.volume_growth_factor = float(os.getenv("VOLUME_GROWTH_FACTOR", "2.0"))
        self.price_surge_percentage = float(os.getenv("PRICE_SURGE_PERCENTAGE", "0.01"))
        
        # Données en temps réel
        self.realtime_data = {}
        self.historical_data = {}
        self.models = {}
        self.last_alerts = {}  # Pour éviter le spam
        
        # WebSocket
        self.ws = None
        self.ws_thread = None
        
        # Instruments à surveiller (paires populaires sur Coinbase)
        self.instruments = [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD',
            'MATIC-USD', 'AVAX-USD', 'LINK-USD', 'UNI-USD', 'ATOM-USD',
            'XLM-USD', 'LTC-USD', 'BCH-USD', 'FIL-USD', 'ALGO-USD'
        ]
        
        logging.info("🚀 Bot de Trading Crypto initialisé avec surveillance Coinbase en temps réel")
        self.send_telegram_message("🤖 Bot de Trading Crypto démarré!\n💹 Surveillance des marchés Coinbase activée")

    def send_telegram_message(self, message):
        """Envoie un message via Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logging.info(f"Message Telegram envoyé: {message[:50]}...")
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
            # Récupération des données historiques
            df = self.get_coinbase_historical_data(product_id)
            if df is None or df.empty:
                return
            
            # Création des caractéristiques
            df_features = self.create_features(df)
            if df_features.empty:
                return
            
            # Entraînement du modèle
            model = self.train_model(df_features, product_id)
            if model is None:
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
            
            # Signal de prédiction
            if abs(prediction) > self.prediction_threshold:
                direction = "📈 ACHAT" if prediction > 0 else "📉 VENTE"
                confidence = min(abs(prediction) * 10000, 100)  # Score de confiance
                signals.append({
                    'type': 'PREDICTION',
                    'direction': direction,
                    'prediction': prediction,
                    'confidence': confidence
                })
            
            # Signal de volume
            volume_ma = latest_data['volume_ma']
            if current_volume > volume_ma * self.volume_growth_factor:
                signals.append({
                    'type': 'VOLUME',
                    'direction': '🔥 VOLUME ÉLEVÉ',
                    'ratio': current_volume / volume_ma
                })
            
            # Signal RSI
            rsi = latest_data['rsi']
            if rsi < 30:
                signals.append({'type': 'RSI', 'direction': '💎 SURVENTE (RSI < 30)', 'value': rsi})
            elif rsi > 70:
                signals.append({'type': 'RSI', 'direction': '⚠️ SURACHAT (RSI > 70)', 'value': rsi})
            
            # Signal Bollinger Bands
            bb_position = latest_data['bb_position']
            if bb_position < 0.1:
                signals.append({'type': 'BB', 'direction': '💎 PROCHE BANDE BASSE', 'value': bb_position})
            elif bb_position > 0.9:
                signals.append({'type': 'BB', 'direction': '⚠️ PROCHE BANDE HAUTE', 'value': bb_position})
            
            # Envoi des alertes
            if signals:
                self.send_trading_alert(product_id, current_price, signals)
                
        except Exception as e:
            logging.error(f"Erreur prédiction {product_id}: {e}")

    def send_trading_alert(self, product_id, price, signals):
        """Envoie une alerte de trading via Telegram"""
        try:
            # Éviter le spam - max 1 alerte par instrument toutes les 5 minutes
            now = time.time()
            if product_id in self.last_alerts:
                if now - self.last_alerts[product_id] < 300:  # 5 minutes
                    return
            
            self.last_alerts[product_id] = now
            
            # Construction du message
            message = f"🚨 <b>ALERTE TRADING</b> 🚨\n"
            message += f"💰 <b>{product_id}</b>\n"
            message += f"💵 Prix: <b>${price:.4f}</b>\n"
            message += f"⏰ {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            for signal in signals:
                if signal['type'] == 'PREDICTION':
                    message += f"{signal['direction']}\n"
                    message += f"📊 Prédiction: {signal['prediction']:.6f}\n"
                    message += f"🎯 Confiance: {signal['confidence']:.1f}%\n\n"
                elif signal['type'] == 'VOLUME':
                    message += f"{signal['direction']}\n"
                    message += f"📈 Ratio: {signal['ratio']:.2f}x\n\n"
                elif signal['type'] == 'RSI':
                    message += f"{signal['direction']}\n"
                    message += f"📊 RSI: {signal['value']:.1f}\n\n"
                elif signal['type'] == 'BB':
                    message += f"{signal['direction']}\n"
                    message += f"📊 Position: {signal['value']:.2f}\n\n"
            
            message += f"🤖 Mode: {self.trading_mode}"
            
            self.send_telegram_message(message)
            logging.info(f"Alerte envoyée pour {product_id}: {len(signals)} signaux")
            
        except Exception as e:
            logging.error(f"Erreur envoi alerte {product_id}: {e}")

    def start_websocket(self):
        """Démarre la connexion WebSocket pour les données temps réel"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data['type'] == 'ticker':
                    product_id = data['product_id']
                    if product_id in self.instruments:
                        # Mise à jour des données temps réel
                        self.realtime_data[product_id] = {
                            'price': float(data['price']),
                            'volume_24h': float(data['volume_24h']),
                            'time': datetime.now()
                        }
                        # Analyse périodique (toutes les 30 secondes max)
                        if product_id in self.historical_data:
                            last_analysis = getattr(self, 'last_analysis', {})
                            now = time.time()
                            if product_id not in last_analysis or now - last_analysis.get(product_id, 0) > 30:
                                last_analysis[product_id] = now
                                self.last_analysis = last_analysis
                                # Analyse basée sur les nouvelles données
                                threading.Thread(
                                    target=self.analyze_realtime_data, 
                                    args=(product_id,)
                                ).start()
            except Exception as e:
                logging.error(f"Erreur WebSocket message: {e}")

        def on_error(ws, error):
            logging.error(f"Erreur WebSocket: {error}")

        def on_close(ws, close_status_code, close_msg):
            logging.warning("WebSocket fermé - tentative de reconnexion...")
            time.sleep(5)
            self.start_websocket()

        def on_open(ws):
            logging.info("✅ WebSocket Coinbase connecté")
            # S'abonner aux tickers
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": self.instruments,
                "channels": ["ticker"]
            }
            ws.send(json.dumps(subscribe_msg))

        # Création et démarrage du WebSocket
        self.ws = websocket.WebSocketApp(
            "wss://ws-feed.exchange.coinbase.com",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def analyze_realtime_data(self, product_id):
        """Analyse les données temps réel"""
        try:
            if product_id not in self.realtime_data or product_id not in self.historical_data:
                return
            
            realtime = self.realtime_data[product_id]
            historical = self.historical_data[product_id]
            
            # Créer une ligne de données simulée basée sur les données temps réel
            latest_row = historical.iloc[-1].copy()
            latest_row['close'] = realtime['price']
            latest_row['volume'] = realtime['volume_24h']
            
            # Recalculer les indicateurs avec le nouveau prix
            # (Simplification - dans un vrai système, on mettrait à jour l'historique complet)
            
            # Faire une prédiction
            self.make_prediction(product_id, latest_row)
            
        except Exception as e:
            logging.error(f"Erreur analyse temps réel {product_id}: {e}")

    def run_analysis_cycle(self):
        """Lance un cycle d'analyse complet"""
        logging.info("🔄 Début du cycle d'analyse...")
        self.send_telegram_message("🔄 Analyse des marchés en cours...")
        
        opportunities = 0
        for instrument in self.instruments:
            try:
                self.analyze_instrument(instrument)
                opportunities += 1
                time.sleep(1)  # Pause pour éviter les limites de taux
            except Exception as e:
                logging.error(f"Erreur analyse {instrument}: {e}")
        
        message = f"✅ Analyse terminée\n📊 {opportunities} instruments analysés\n🤖 Surveillance temps réel active"
        self.send_telegram_message(message)
        logging.info(f"Cycle d'analyse terminé - {opportunities} instruments")

    def start_scheduled_analysis(self):
        """Programme les analyses périodiques"""
        # Analyse complète toutes les heures
        schedule.every().hour.do(self.run_analysis_cycle)
        
        # Message de statut toutes les 4 heures
        schedule.every(4).hours.do(self.send_status_message)
        
        logging.info("📅 Analyses programmées: toutes les heures")

    def send_status_message(self):
        """Envoie un message de statut"""
        active_instruments = len(self.realtime_data)
        models_count = len(self.models)
        
        message = f"🤖 <b>BOT STATUS</b>\n"
        message += f"📡 Instruments actifs: {active_instruments}\n"
        message += f"🧠 Modèles entraînés: {models_count}\n"
        message += f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        message += f"✅ Surveillance active"
        
        self.send_telegram_message(message)

    def run(self):
        """Démarre le bot"""
        try:
            # Message de démarrage
            self.send_telegram_message("🚀 <b>BOT DE TRADING DÉMARRÉ</b>\n📊 Initialisation en cours...")
            
            # Analyse initiale
            self.run_analysis_cycle()
            
            # Démarrage du WebSocket pour données temps réel
            self.start_websocket()
            
            # Programmation des tâches
            self.start_scheduled_analysis()
            
            # Message de confirmation
            self.send_telegram_message("✅ <b>BOT OPÉRATIONNEL</b>\n🔴 Surveillance temps réel active\n📱 Vous recevrez les alertes ici")
            
            # Boucle principale
            logging.info("🎯 Bot opérationnel - En attente des signaux...")
            while True:
                schedule.run_pending()
                time.sleep(60)  # Vérification chaque minute
                
        except KeyboardInterrupt:
            logging.info("Arrêt du bot demandé par l'utilisateur")
            self.send_telegram_message("🛑 Bot arrêté par l'utilisateur")
        except Exception as e:
            logging.error(f"Erreur critique: {e}")
            self.send_telegram_message(f"❌ Erreur critique: {e}")
        finally:
            if self.ws:
                self.ws.close()

def main():
    bot = CoinbaseRealtimeBot()
    bot.run()

if __name__ == "__main__":
    main()
