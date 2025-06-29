# 🤖 Bot de Trading Crypto - Coinbase + Telegram

## 📋 Installation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Créer votre bot Telegram

1. Ouvrez Telegram et cherchez **@BotFather**
2. Tapez `/newbot`
3. Donnez un nom à votre bot (ex: "Mon Bot Trading")
4. Donnez un username (ex: "mon_trading_bot")
5. **Copiez le token** fourni par BotFather

### 3. Obtenir votre Chat ID

1. Envoyez un message à votre bot
2. Allez sur: `https://api.telegram.org/bot<VOTRE_TOKEN>/getUpdates`
3. Cherchez votre **chat_id** dans la réponse

### 4. Configuration

1. Copiez `.env.example` vers `.env`
2. Remplissez au minimum:
   ```
   TELEGRAM_BOT_TOKEN=votre_token_du_botfather
   TELEGRAM_CHAT_ID=votre_chat_id
   ```

### 5. Lancer le bot
```bash
python coinbase_telegram_bot.py
```

## ⚙️ Configuration avancée

### Variables d'environnement (.env)

```env
# OBLIGATOIRE - Telegram
TELEGRAM_BOT_TOKEN=1234567890:ABCDEF...
TELEGRAM_CHAT_ID=123456789

# OPTIONNEL - Coinbase Pro (pour trading réel)
COINBASE_API_KEY=votre_clé
COINBASE_API_SECRET=votre_secret
COINBASE_PASSPHRASE=votre_passphrase

# PARAMÈTRES DU BOT
TRADING_MODE=SIMULATION
PREDICTION_THRESHOLD=0.00002
VOLUME_GROWTH_FACTOR=2.0
PRICE_SURGE_PERCENTAGE=0.01
```

## 🎯 Fonctionnalités

### ✅ Surveillance temps réel
- **WebSocket Coinbase**: Données en temps réel
- **15 cryptos populaires**: BTC, ETH, ADA, SOL, etc.
- **Analyses automatiques**: Toutes les heures

### 📊 Signaux de trading
- **Prédictions IA**: Random Forest avec 11 indicateurs
- **RSI**: Détection survente/surachat
- **Volume**: Alertes sur volumes élevés
- **Bollinger Bands**: Signaux de retournement
- **MACD**: Croisements de moyennes

### 📱 Notifications Telegram
- **Alertes instantanées**: Dès qu'un signal est détecté
- **Anti-spam**: Max 1 alerte/5min par crypto
- **Messages formatés**: Emojis et détails techniques
- **Statut périodique**: État du bot toutes les 4h

## 🔧 Utilisation

### Commandes du bot
- Le bot s'auto-gère, pas de commandes manuelles
- Messages automatiques pour signaux et statut
- Logs détaillés dans `crypto_trading_bot.log`

### Types d'alertes reçues
```
🚨 ALERTE TRADING 🚨
💰 BTC-USD
💵 Prix: $45,234.56
⏰ 14:32:15

📈 ACHAT
📊 Prédiction: 0.003456
🎯 Confiance: 89.2%

🔥 VOLUME ÉLEVÉ
📈 Ratio: 3.4x

🤖 Mode: SIMULATION
```

## 🛠️ Personnalisation

### Modifier les cryptos surveillées
Dans `coinbase_telegram_bot.py`, ligne 58:
```python
self.instruments = [
    'BTC-USD', 'ETH-USD', 'ADA-USD',
    # Ajoutez vos cryptos ici
]
```

### Ajuster la sensibilité
Dans `.env`:
```env
PREDICTION_THRESHOLD=0.00001  # Plus bas = plus sensible
VOLUME_GROWTH_FACTOR=1.5      # Plus bas = plus d'alertes volume
```

## 🚨 Mode Trading Réel

⚠️ **ATTENTION**: Par défaut en SIMULATION

Pour activer le trading réel:
1. Configurez les clés Coinbase Pro dans `.env`
2. Changez `TRADING_MODE=REAL`
3. **Testez d'abord avec de petites sommes**

## 📝 Logs et Debug

- **Fichier log**: `crypto_trading_bot.log`
- **Console**: Affichage en temps réel
- **Telegram**: Messages d'erreur critiques

## 🔒 Sécurité

- ✅ Variables d'environnement pour clés
- ✅ Pas de clés dans le code
- ✅ Gestion d'erreurs robuste
- ✅ Limites de taux API respectées

## 🆘 Dépannage

### Bot ne démarre pas
1. Vérifiez le token Telegram
2. Vérifiez le chat_id
3. Installez les dépendances: `pip install -r requirements.txt`

### Pas d'alertes reçues
1. Augmentez la sensibilité dans `.env`
2. Vérifiez les logs pour erreurs
3. Testez avec `/start` dans Telegram

### Erreurs WebSocket
- Normal, reconnexion automatique
- Vérifiez votre connexion internet
- Redémarrez le bot si persistant

## 📞 Support

Pour problèmes ou questions:
1. Vérifiez les logs
2. Testez la configuration
3. Consultez la documentation Telegram Bot API
