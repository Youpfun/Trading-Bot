import requests
import time
import hashlib
import hmac
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from plyer import notification
from dotenv import load_dotenv # Importe la fonction pour charger les variables d'environnement
import logging
import smtplib
import ssl
from email.mime.text import MIMEText

# --- 1. Configuration du Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("trading_bot.log"), # Log dans un fichier
                        logging.StreamHandler() # Log sur la console
                    ])

# --- 2. Chargement des variables d'environnement pour la sécurité ---
load_dotenv() # Charge les variables du fichier .env

# Remplacez par vos clés API via des variables d'environnement (correctement)
API_KEY = os.getenv("CRYPTO_COM_API_KEY")
API_SECRET = os.getenv("CRYPTO_COM_API_SECRET")

# Informations pour l'envoi d'e-mails
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = os.getenv("EMAIL_SMTP_PORT", 587) # Par défaut à 587 pour TLS/STARTTLS

# Vérification que les clés API et les infos email sont chargées
if not API_KEY or not API_SECRET:
    logging.error("Les clés API n'ont pas été chargées depuis les variables d'environnement. Veuillez vérifier votre fichier .env ou vos variables système.")
    exit("Erreur critique : Clés API manquantes. Arrêt du programme.")
if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT or not EMAIL_SMTP_SERVER:
    logging.warning("Les informations d'authentification pour l'e-mail ne sont pas complètement configurées. Les e-mails ne seront pas envoyés.")

# CORRECTION CLÉ : Changement de l'URL de base pour correspondre à l'API Exchange v1 pour la plupart des endpoints
BASE_URL = "https://api.crypto.com/exchange/v1"

def generate_signature(method, params, api_key, api_secret):
    nonce = int(time.time() * 1000)
    # Les paramètres doivent être triés pour la signature
    if params:
        # Assurez-vous que les valeurs des paramètres sont converties en string pour le param_str
        param_str = ''.join(f'{k}{str(params[k])}' for k in sorted(params))
    else:
        param_str = ''

    # L'ordre des éléments pour la signature Crypto.com V2 est important
    # C'est : method + nonce + api_key + param_str
    payload_to_sign = f"{method}{nonce}{api_key}{param_str}"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        payload_to_sign.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return {
        "id": 1,
        "method": method,
        "api_key": api_key,
        "params": params,
        "nonce": nonce,
        "sig": signature
    }

def get_historical_data(instrument_name, timeframe, limit):
    """Récupère les données historiques des chandeliers."""
    method = "public/get-candlestick"
    params = {
        "instrument_name": instrument_name,
        "timeframe": timeframe,
        "limit": limit
    }
    # logging.info(f"Récupération des données historiques pour {instrument_name}, timeframe: {timeframe}, limit: {limit}")
    try:
        response = requests.get(f"{BASE_URL}/{method}", params=params, timeout=10)
        response.raise_for_status() # Lève une exception pour les codes d'état HTTP 4xx/5xx
        json_response = response.json()
        if json_response.get('code') == 0 and json_response.get('result') and json_response['result'].get('data'):
            return json_response
        else:
            logging.error(f"Réponse invalide ou vide de {method} pour {instrument_name}: {json_response}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de la récupération des données historiques pour {instrument_name}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Erreur de décodage JSON pour {method} pour {instrument_name}: {e}. Réponse brute: {response.text}")
        return None

def get_current_price(instrument_name):
    """Récupère le prix actuel du ticker pour un instrument donné."""
    method = "public/get-ticker" # L'endpoint pour un seul ticker dans la v1 est public/get-ticker (singulier)
    params = {
        "instrument_name": instrument_name
    }
    # logging.info(f"Récupération du prix actuel pour {instrument_name}")
    try:
        response = requests.get(f"{BASE_URL}/{method}", params=params, timeout=5)
        response.raise_for_status()
        json_response = response.json()
        # Vérification robuste de la structure de la réponse de 'public/get-ticker'
        # La clé pour le dernier prix est 'k' dans la réponse que vous avez fournie.
        if json_response.get('code') == 0 and json_response.get('result') and json_response['result'].get('data') and len(json_response['result']['data']) > 0:
            return json_response
        else:
            logging.error(f"Réponse invalide ou vide de {method} pour {instrument_name}: {json_response}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de la récupération du prix actuel pour {instrument_name}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Erreur de décodage JSON pour {method} pour {instrument_name}: {e}. Réponse brute: {response.text}")
        return None

def get_all_instruments():
    """Récupère tous les détails des instruments disponibles."""
    method = "public/get-instruments"
    # L'endpoint public/get-instruments ne prend PAS instrument_name comme paramètre.
    # Il retourne TOUS les instruments.
    params = {}
    logging.info("Récupération de tous les détails des instruments disponibles.")
    try:
        response = requests.get(f"{BASE_URL}/{method}", params=params, timeout=5)
        response.raise_for_status()
        json_response = response.json()
        # CORRECTION ICI : L'API retourne les instruments sous la clé 'data' et non 'instruments' dans le 'result'
        if json_response.get('code') == 0 and json_response.get('result') and json_response['result'].get('data') and len(json_response['result']['data']) > 0:
            return json_response['result']['data'] # Changement de 'instruments' à 'data'
        else:
            logging.error(f"Réponse invalide ou vide de {method}: {json_response}")
            return [] # Retourne une liste vide en cas d'échec
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de la récupération de tous les instruments (HTTP/Connexion): {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Erreur de décodage JSON pour {method}: {e}. Réponse brute: {response.text}")
        return []

def preprocess_data(data):
    if not data or data.get('code') != 0 or not data['result']['data']:
        logging.warning("Données historiques invalides ou vides reçues lors du prétraitement.")
        return pd.DataFrame() # Retourne un DataFrame vide

    df = pd.DataFrame(data['result']['data'])
    # Renommer les colonnes pour la clarté et la cohérence
    df.rename(columns={'t': 'time', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']] # Assurer l'ordre des colonnes
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True) # Assurer que les données sont triées par temps
    return df

def create_features(df, window_size):
    if df.empty or len(df) < window_size:
        # logging.warning(f"DataFrame trop court pour créer des caractéristiques avec window_size={window_size}. Taille actuelle: {len(df)}")
        return pd.DataFrame()

    df['returns'] = df['close'].pct_change()
    df['ma'] = df['close'].rolling(window=window_size).mean()
    df['std'] = df['close'].rolling(window=window_size).std()
    # Ajout de caractéristiques supplémentaires pour un modèle potentiellement meilleur
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

    # S'assurer que les caractéristiques requises pour X sont créées avant dropna
    required_features = ['ma', 'std', 'returns', 'rsi', 'macd', 'macd_signal', 'macd_hist']
    for col in required_features:
        if col not in df.columns:
            logging.error(f"Caractéristique '{col}' manquante après création des caractéristiques.")
            return pd.DataFrame() # Retourne un DataFrame vide si une caractéristique manque

    df.dropna(inplace=True) # Supprime les lignes avec des valeurs NaN créées par les fenêtres glissantes ou pct_change
    return df

def calculate_rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    # Gérer la division par zéro si ema_down est 0
    rs = ema_up / ema_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def train_model(X, y):
    if len(X) < 2: # Nécessite au moins 2 échantillons pour le split
        logging.warning("Pas assez de données pour entraîner le modèle. X a moins de 2 échantillons.")
        return None

    # Ajustement pour s'assurer qu'il y a suffisamment de données pour le test_size
    if len(X) < (1 / 0.2): # Si test_size=0.2, besoin d'au moins 5 échantillons
        logging.warning(f"Pas assez de données pour le train_test_split (nécessite > {int(1/0.2)} échantillons). X a {len(X)} échantillons.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Ajout de shuffle=False pour les séries temporelles
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 pour utiliser tous les cœurs
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Mean Squared Error du modèle : {mse:.6f}")
    return model

def place_order(instrument_name, side, quantity, price):
    method = "private/create-order"
    # Convertir la quantité et le prix en chaînes avec la précision nécessaire
    # Assurez-vous d'utiliser une précision raisonnable pour les strings (ex: 8 décimales pour quantité, 2 pour prix)
    formatted_quantity = f"{quantity:.8f}".rstrip('0').rstrip('.') # Supprime les zéros de fin inutiles
    formatted_price = f"{price:.8f}".rstrip('0').rstrip('.') # Utilisez 8 décimales pour être sûr, l'API gérera

    params = {
        "instrument_name": instrument_name,
        "side": side,
        "type": "LIMIT",
        "price": formatted_price,      # Le prix est requis pour un ordre LIMIT
        "quantity": formatted_quantity, # La quantité est requise pour un ordre LIMIT
        "client_oid": f"bot_{int(time.time() * 1000)}_{os.urandom(4).hex()}" # ID client unique avec plus de randomness
    }
    logging.info(f"Tentative de placer un ordre : {side} {formatted_quantity} {instrument_name} @ {formatted_price}")
    try:
        payload = generate_signature(method, params, API_KEY, API_SECRET)
        response = requests.post(f"{BASE_URL}/{method}", json=payload, timeout=10)
        response.raise_for_status()
        json_response = response.json()
        logging.info(f"Réponse de l'ordre : {json_response}")
        return json_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors du placement de l'ordre : {e}")
        return {"code": -1, "message": str(e)}
    except json.JSONDecodeError as e:
        logging.error(f"Erreur de décodage JSON après le placement de l'ordre : {e}. Réponse brute: {response.text}")
        return {"code": -1, "message": "Erreur de décodage JSON"}


def notify(message):
    try:
        notification.notify(
            title="Alerte Trading Crypto Bot",
            message=message,
            app_name="CryptoBot",
            timeout=10
        )
    except Exception as e:
        logging.error(f"Impossible d'envoyer la notification de bureau : {e}")

def send_email_notification(subject, body):
    """Envoie une notification par e-mail."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT or not EMAIL_SMTP_SERVER:
        logging.warning("Informations e-mail incomplètes. E-mail non envoyé.")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, int(EMAIL_SMTP_PORT)) as server:
            server.starttls(context=context)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info(f"E-mail envoyé avec succès à {EMAIL_RECIPIENT} : {subject}")
    except Exception as e:
        logging.error(f"Échec de l'envoi de l'e-mail : {e}")

def detect_emerging_signals(df, instrument_name, current_price, current_volume,
                            volume_growth_factor=2.0, price_surge_percentage=0.05, price_surge_period=5):
    """
    Détecte les signaux de cryptos "émergentes" basés sur la croissance du volume
    et les fortes augmentations de prix.
    Retourne une liste de dictionnaires d'opportunités d'émergence.
    """
    signals = []

    if df.empty or len(df) < max(20, price_surge_period + 1): # Besoin de suffisamment de données pour les calculs
        return signals

    # Signal 1: Croissance significative du volume
    # Comparer le dernier volume avec la moyenne des volumes précédents (ex: sur 20 périodes)
    historical_volume_avg = df['volume'].iloc[:-1].rolling(window=min(20, len(df)-1)).mean().iloc[-1]
    if pd.isna(historical_volume_avg) or historical_volume_avg == 0:
        historical_volume_avg = df['volume'].mean() # Fallback si pas assez de données pour le rolling mean
        if historical_volume_avg == 0: # Évite la division par zéro
            historical_volume_avg = 1 # Juste pour éviter l'erreur

    if current_volume > historical_volume_avg * volume_growth_factor:
        signal_msg = (f"Forte augmentation du volume! Volume actuel: {current_volume:.2f}, "
                      f"Moyenne historique: {historical_volume_avg:.2f} (Facteur: {current_volume/historical_volume_avg:.2f}x)")
        signals.append({
            'type': 'Émergence Volume',
            'details': signal_msg,
            'strength': current_volume / historical_volume_avg # Force du signal basée sur le facteur de croissance
        })
        logging.info(f"Signal d'émergence détecté (Volume) pour {instrument_name}: {signal_msg}")
        notify(f"Alerte ÉMERGENCE: {instrument_name} - Volume en hausse !")


    # Signal 2: Forte augmentation du prix sur une courte période
    # Calculer le changement de prix sur les 'price_surge_period' dernières bougies
    if len(df) >= price_surge_period + 1:
        price_start = df['close'].iloc[-(price_surge_period + 1)]
        price_end = df['close'].iloc[-1]
        if price_start > 0: # Évite la division par zéro
            price_change = (price_end - price_start) / price_start
            if price_change > price_surge_percentage:
                signal_msg = (f"Forte hausse de prix sur {price_surge_period} périodes! "
                              f"Changement: {price_change:.2%}")
                signals.append({
                    'type': 'Émergence Prix',
                    'details': signal_msg,
                    'strength': price_change # Force du signal basée sur le pourcentage de changement
                })
                logging.info(f"Signal d'émergence détecté (Prix) pour {instrument_name}: {signal_msg}")
                notify(f"Alerte ÉMERGENCE: {instrument_name} - Prix en forte hausse !")
    return signals

def send_summary_email(opportunities_list, total_instruments_scanned):
    """Envoie un e-mail de résumé des meilleures opportunités détectées."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT or not EMAIL_SMTP_SERVER:
        logging.warning("Informations e-mail incomplètes. E-mail de résumé non envoyé.")
        return

    # Sépare et trie les opportunités
    trading_opportunities = sorted([
        op for op in opportunities_list if op['category'] == 'Prédiction Trading'
    ], key=lambda x: abs(x['prediction']), reverse=True) # Tri par magnitude de la prédiction

    emerging_opportunities = sorted([
        op for op in opportunities_list if op['category'] == 'Émergence'
    ], key=lambda x: x['strength'], reverse=True) # Tri par force du signal d'émergence

    email_body = "Bonjour,\n\nVoici le résumé des opportunités détectées lors du dernier cycle de balayage des marchés cryptos.\n\n"

    if trading_opportunities:
        email_body += "--- Top 5 des Prédictions de Trading ---\n"
        for i, op in enumerate(trading_opportunities[:5]): # Top 5 seulement
            email_body += (f"{i+1}. {op['instrument']} ({op['type']}): "
                           f"Prédiction de retour = {op['prediction']:.6f} | "
                           f"Prix actuel = {op['price']:.2f} | "
                           f"Quantité (simulée) = {op['quantity']:.4f} | " # Afficher la quantité
                           f"Détails: {op['details']}\n")
        email_body += "\n"
    else:
        email_body += "Aucune prédiction de trading significative détectée ce cycle selon les seuils.\n\n"

    if emerging_opportunities:
        email_body += "--- Top 5 des Signaux d'Émergence ---\n"
        for i, op in enumerate(emerging_opportunities[:5]): # Top 5 seulement
            email_body += (f"{i+1}. {op['instrument']} ({op['type']}): "
                           f"Force du signal: {op['strength']:.2f} | "
                           f"Prix actuel = {op['price']:.2f} | "
                           f"Détails: {op['details']}\n")
        email_body += "\n"
    else:
        email_body += "Aucun signal d'émergence détecté ce cycle selon les critères.\n\n"

    email_body += f"Total d'instruments analysés ce cycle: {total_instruments_scanned}.\n"
    email_body += f"Total d'opportunités significatives détectées (incluant trading et émergence): {len(opportunities_list)}.\n\n"
    email_body += "Cordialement,\nVotre Crypto Trading Bot"

    subject = f"Résumé Crypto Bot ({len(opportunities_list)} Opportunités)"
    if not opportunities_list:
        subject = "Résumé Crypto Bot - Aucune Opportunité Détectée"

    send_email_notification(subject, email_body)
    logging.info("E-mail de résumé envoyé.")


def main():
    TIMEFRAME = "15m"
    LIMIT = 200 # Augmenté de 100 à 200 pour plus de données historiques
    PREDICTION_THRESHOLD = 0.00002 # Abaissé de 0.00005 à 0.00002 pour plus de sensibilité
    DEFAULT_TRADING_QUANTITY = 0.0001

    QUOTE_CURRENCIES_TO_MONITOR = ['USDT', 'USDC', 'BTC', 'EURA', 'EUR']
    VOLUME_GROWTH_FACTOR = 2.0 # Abaissé de 3.0 à 2.0 pour plus de sensibilité
    PRICE_SURGE_PERCENTAGE = 0.01 # Abaissé de 0.03 à 0.01 pour plus de sensibilité
    PRICE_SURGE_PERIOD = 5

    # Ligne pour vérifier l'envoi de mail au démarrage du bot
    if EMAIL_SENDER and EMAIL_RECIPIENT and EMAIL_PASSWORD and EMAIL_SMTP_SERVER and EMAIL_SMTP_PORT:
        send_email_notification("TEST: Démarrage du Crypto Trading Bot", "Le bot de trading a démarré et tente d'envoyer un e-mail de test. Si vous recevez cet e-mail, la configuration est correcte.")
        logging.info("E-mail de test envoyé au démarrage du bot.")
    else:
        logging.warning("Impossible d'envoyer l'e-mail de test : les informations d'e-mail sont incomplètes ou incorrectes dans le fichier .env. Veuillez vérifier EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT.")


    all_instruments_data = get_all_instruments()
    if not all_instruments_data:
        logging.critical("Impossible de récupérer la liste de tous les instruments. Arrêt du programme.")
        notify("Erreur critique: Liste des instruments non disponible. Arrêt.")
        send_email_notification("Erreur Critique Bot Trading", "Impossible de récupérer la liste des instruments. Arrêt du bot.")
        return

    tradable_instruments = [
        instr for instr in all_instruments_data
        if instr.get('tradable') and
           instr.get('inst_type') == 'CCY_PAIR' and
           instr.get('quote_ccy') in QUOTE_CURRENCIES_TO_MONITOR
    ]
    logging.info(f"Nombre d'instruments tradables et filtrés trouvés : {len(tradable_instruments)}")
    if not tradable_instruments:
        logging.warning("Aucun instrument tradable trouvé après le filtrage. Vérifiez vos critères de filtrage ou l'API.")
        notify("Avertissement: Aucun instrument tradable trouvé.")
        send_email_notification("Avertissement Crypto Bot", "Aucun instrument tradable trouvé après le filtrage initial.")


    while True:
        cycle_opportunities = [] # Liste pour stocker toutes les opportunités détectées dans ce cycle
        total_analyzed_in_cycle = 0

        for instrument_data in tradable_instruments:
            instrument_name = instrument_data.get('symbol')
            if not instrument_name:
                logging.warning(f"Instrument sans 'symbol' trouvé: {instrument_data}. Ignoré.")
                continue

            total_analyzed_in_cycle += 1
            logging.info(f"--- Démarrage de l'analyse pour l'instrument : {instrument_name} ({total_analyzed_in_cycle}/{len(tradable_instruments)}) ---")


            try:
                price_decimals = instrument_data.get('quote_decimals', 2)
                quantity_decimals = instrument_data.get('quantity_decimals', 8)
                min_quantity = float(instrument_data.get('qty_tick_size', 0.00001))
                min_notional = 10 # Default or adjust based on your strategy


                # --- 1. Collecte des données historiques ---
                historical_data = get_historical_data(instrument_name, TIMEFRAME, LIMIT)
                if not historical_data:
                    logging.warning(f"Échec de la récupération des données historiques pour {instrument_name}. Passer au suivant.")
                    time.sleep(0.1) # Petite pause pour éviter la surcharge API
                    continue

                df = preprocess_data(historical_data)
                if df.empty:
                    logging.warning(f"Le DataFrame traité est vide pour {instrument_name}. Passer au suivant.")
                    time.sleep(0.1)
                    continue

                # --- 2. Création des caractéristiques ---
                df_features = create_features(df, window_size=30)
                expected_features = ['ma', 'std', 'returns', 'rsi', 'macd', 'macd_signal', 'macd_hist']
                if df_features.empty or len(df_features) < 30 or any(col not in df_features.columns for col in expected_features):
                    logging.warning(f"DataFrame trop court ou caractéristiques manquantes pour {instrument_name}. Passer au suivant. Taille: {len(df_features)}")
                    time.sleep(0.1)
                    continue

                # --- Détection des signaux d'émergence AVANT l'entraînement du modèle ---
                current_volume_for_emerging = df['volume'].iloc[-1] if not df['volume'].empty else 0
                current_price_for_emerging = df['close'].iloc[-1] if not df['close'].empty else 0

                emerging_signals_detected = detect_emerging_signals(df, instrument_name, current_price_for_emerging, current_volume_for_emerging,
                                                                    volume_growth_factor=VOLUME_GROWTH_FACTOR,
                                                                    price_surge_percentage=PRICE_SURGE_PERCENTAGE,
                                                                    price_surge_period=PRICE_SURGE_PERIOD)
                for signal_info in emerging_signals_detected:
                    cycle_opportunities.append({
                        'instrument': instrument_name,
                        'category': 'Émergence',
                        'type': signal_info['type'],
                        'strength': signal_info['strength'],
                        'details': signal_info['details'],
                        'price': current_price_for_emerging # Ajouter le prix actuel pour le contexte
                    })


                # --- 3. Préparation des données pour l'entraînement ---
                X = df_features[expected_features]
                y = df_features['returns']

                # --- 4. Entraînement du modèle ---
                model = train_model(X, y)
                if model is None:
                    logging.warning(f"Modèle non entraîné pour {instrument_name}. Passer au suivant.")
                    time.sleep(0.1)
                    continue

                # --- 5. Utilisation du modèle pour prédire les retours ---
                if X.empty:
                    logging.warning(f"Le DataFrame X est vide pour {instrument_name}, impossible de faire une prédiction. Passer au suivant.")
                    time.sleep(0.1)
                    continue

                latest_data_df = pd.DataFrame(X.iloc[-1].values.reshape(1, -1), columns=X.columns)
                prediction = model.predict(latest_data_df)[0]
                logging.info(f"Prédiction de retour pour {instrument_name} : {prediction:.6f}")

                # --- 6. Prise de décision de trading ---
                current_price_response = get_current_price(instrument_name)
                current_price = None
                if current_price_response:
                    try:
                        current_price = float(current_price_response['result']['data'][0]['k'])
                    except (KeyError, IndexError, TypeError) as e:
                        logging.error(f"Impossible d'extraire le prix de la réponse de l'API Ticker pour {instrument_name}: {e}. Réponse: {current_price_response}")
                        notify(f"Avertissement: Impossible d'obtenir prix pour {instrument_name}. Ordre non placé.")
                        time.sleep(0.1)
                        continue
                else:
                    logging.warning(f"La fonction get_current_price a retourné None pour {instrument_name}. Impossible d'obtenir le prix actuel.")
                    notify(f"Avertissement: Impossible d'obtenir prix pour {instrument_name}. Ordre non placé.")
                    time.sleep(0.1)
                    continue

                # --- Gestion de la quantité et du prix selon les détails de l'instrument ---
                adjusted_price = round(current_price, price_decimals)
                adjusted_quantity = DEFAULT_TRADING_QUANTITY

                target_notional = max(min_notional, DEFAULT_TRADING_QUANTITY * adjusted_price)
                if adjusted_price > 0:
                    calculated_quantity = target_notional / adjusted_price
                else:
                    calculated_quantity = DEFAULT_TRADING_QUANTITY

                adjusted_quantity = round(max(min_quantity, calculated_quantity), quantity_decimals)


                if adjusted_quantity < min_quantity:
                    logging.warning(f"La quantité ajustée pour {instrument_name} ({adjusted_quantity}) est toujours inférieure à la quantité minimale ({min_quantity}).")
                    notify(f"Avertissement: Quantité insuffisante pour {instrument_name}. Ordre non placé.")
                    # Pas d'e-mail ici pour éviter de spammer pour chaque instrument non conforme.
                    time.sleep(0.1)
                    continue

                current_notional_after_adjustment = adjusted_quantity * adjusted_price
                if current_notional_after_adjustment < min_notional:
                    logging.warning(f"La valeur notionnelle de l'ordre pour {instrument_name} ({current_notional_after_adjustment:.2f}) est inférieure au minimum ({min_notional}). Ordre non placé.")
                    notify(f"Avertissement: Valeur de l'ordre trop faible pour {instrument_name}. Ordre non placé.")
                    # Pas d'e-mail ici non plus.
                    time.sleep(0.1)
                    continue


                if prediction > PREDICTION_THRESHOLD:
                    logging.info(f"Prédiction de hausse ({prediction:.6f} > {PREDICTION_THRESHOLD}) pour {instrument_name}, ajout à la liste des opportunités.")
                    alert_message = f"Prédiction de hausse pour {instrument_name} ! Achat de {adjusted_quantity} @ {adjusted_price} USD."
                    notify(alert_message) # Notification de bureau immédiate
                    # Stocker l'opportunité pour le résumé par e-mail
                    cycle_opportunities.append({
                        'instrument': instrument_name,
                        'category': 'Prédiction Trading',
                        'type': 'ACHAT',
                        'prediction': prediction,
                        'price': adjusted_price,
                        'quantity': adjusted_quantity,
                        'details': alert_message
                    })
                    logging.info(f"Ordre d'achat SIMULÉ pour {instrument_name} car le trading est désactivé par défaut.")
                elif prediction < -PREDICTION_THRESHOLD:
                    logging.info(f"Prédiction de baisse ({prediction:.6f} < {-PREDICTION_THRESHOLD}) pour {instrument_name}, ajout à la liste des opportunités.")
                    alert_message = f"Prédiction de baisse pour {instrument_name} ! Vente de {adjusted_quantity} @ {adjusted_price} USD."
                    notify(alert_message) # Notification de bureau immédiate
                    # Stocker l'opportunité pour le résumé par e-mail
                    cycle_opportunities.append({
                        'instrument': instrument_name,
                        'category': 'Prédiction Trading',
                        'type': 'VENTE',
                        'prediction': prediction,
                        'price': adjusted_price,
                        'quantity': adjusted_quantity,
                        'details': alert_message
                    })
                    logging.info(f"Ordre de vente SIMULÉ pour {instrument_name} car le trading est désactivé par default.")
                else:
                    logging.info(f"Prédiction neutre ({prediction:.6f}) pour {instrument_name}. Aucun ordre placé ni opportunité ajoutée.")


            except requests.exceptions.Timeout:
                logging.error(f"Timeout de la requête API pour {instrument_name}. Le serveur ne répond pas à temps.")
                notify(f"Erreur API: Timeout pour {instrument_name}.")
                time.sleep(1)
            except requests.exceptions.ConnectionError as ce:
                logging.error(f"Erreur de connexion réseau pour {instrument_name}: {ce}. Vérifiez votre connexion Internet.")
                notify(f"Erreur réseau: {instrument_name}.")
                time.sleep(1)
            except requests.exceptions.HTTPError as he:
                logging.error(f"Erreur HTTP de l'API pour {instrument_name}: {he}. Code: {he.response.status_code}. Réponse: {he.response.text}")
                notify(f"Erreur HTTP API: {he.response.status_code} pour {instrument_name}.")
                time.sleep(1)
            except json.JSONDecodeError as json_e:
                logging.error(f"Erreur de décodage JSON de la réponse API pour {instrument_name}: {json_e}. Réponse inattendue.")
                notify(f"Erreur JSON API: {instrument_name}.")
                time.sleep(1)
            except pd.errors.EmptyDataError:
                logging.error(f"Erreur Pandas : Le DataFrame est vide pour {instrument_name}.")
                notify(f"Erreur Pandas: {instrument_name}.")
                time.sleep(1)
            except Exception as e:
                logging.exception(f"Une erreur inattendue et non gérée s'est produite pour {instrument_name}: {e}")
                notify(f"Erreur critique bot: {instrument_name}: {e}")
                send_email_notification(f"ERREUR CRITIQUE Crypto Bot: {instrument_name}", f"Une erreur non gérée s'est produite: {e}")
                time.sleep(5) # Pause plus longue en cas d'erreur inattendue

            # Petite pause entre chaque instrument pour respecter les limites de débit de l'API
            time.sleep(0.5)

        # --- Envoi du résumé par e-mail après le balayage de tous les instruments ---
        send_summary_email(cycle_opportunities, total_analyzed_in_cycle)
        logging.info(f"Fin du cycle de balayage des marchés. {len(cycle_opportunities)} opportunités détectées. Pause de 15 minutes avant le prochain balayage complet.")
        time.sleep(900) # CHANGEMENT ICI : Pause de 15 minutes (15 * 60 = 900 secondes)

if __name__ == "__main__":
    main()
