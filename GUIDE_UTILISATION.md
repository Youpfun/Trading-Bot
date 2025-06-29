# 🤖 GUIDE D'UTILISATION - Bot de Trading Telegram

## ✅ Votre Chat ID est déjà configuré : `5038189418`

## 🔑 DERNIÈRE ÉTAPE : Configurer votre Token Telegram

### 1. Obtenez votre Token Telegram
1. Ouvrez Telegram
2. Cherchez `@BotFather`
3. Envoyez `/newbot`
4. Suivez les instructions
5. **Copiez le token** fourni (format: `1234567890:ABCDEF...`)

### 2. Modifiez le fichier .env
Ouvrez le fichier `.env` et remplacez :
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_from_botfather
```

Par votre vrai token :
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCDEF_votre_vrai_token_ici
```

### 3. Lancez le bot
```bash
python coinbase_telegram_bot.py
```

## 🎯 Fonctionnalités du Bot

### ✅ Ce que fait le bot :
- **Surveillance temps réel** des cryptos sur Coinbase
- **Analyse IA** avec Random Forest (11 indicateurs techniques)
- **Alertes Telegram** instantanées
- **15 cryptos populaires** : BTC, ETH, ADA, SOL, etc.
- **Mode simulation** par défaut (sécurisé)

### 📊 Types d'alertes que vous recevrez :

#### 🔮 Prédictions IA
```
🚨 ALERTE TRADING 🚨
💰 BTC-USD
💵 Prix: $45,234.56
⏰ 14:32:15

📈 ACHAT
📊 Prédiction: 0.003456
🎯 Confiance: 89.2%
```

#### 🔥 Volume élevé
```
🔥 VOLUME ÉLEVÉ
📈 Ratio: 3.4x
💰 BTC-USD
```

#### 📈 Signaux techniques
- **RSI** : Survente/Surachat
- **MACD** : Croisements
- **Bollinger Bands** : Retournements

### ⚙️ Paramètres (dans .env)
```env
# Sensibilité des alertes
PREDICTION_THRESHOLD=0.00002    # Plus bas = plus d'alertes
VOLUME_GROWTH_FACTOR=2.0        # Seuil volume élevé
PRICE_SURGE_PERCENTAGE=0.01     # Seuil hausse prix

# Mode trading
TRADING_MODE=SIMULATION         # SIMULATION ou REAL
```

## 🛠️ Dépannage

### ❌ "Token Telegram invalide"
- Vérifiez que votre token est correct
- Testez avec : `https://api.telegram.org/bot<TOKEN>/getMe`

### ❌ "Chat ID manquant"
- Votre Chat ID est déjà configuré : `5038189418`
- Si problème, utilisez `python get_chat_id.py`

### ❌ "Erreurs d'encodage"
- Normal sous Windows avec les emojis
- Le bot fonctionne quand même

### ❌ "Pas d'alertes reçues"
- Vérifiez que le bot tourne
- Augmentez la sensibilité dans `.env`
- Les analyses se font toutes les heures

## 📱 Commandes Telegram

Le bot fonctionne automatiquement, pas de commandes manuelles nécessaires.

### Messages automatiques :
- **Démarrage** : Confirmation que le bot est lancé
- **Alertes** : Signaux de trading détectés
- **Statut** : État du bot toutes les 4h

## 🔒 Sécurité

- ✅ Mode SIMULATION par défaut
- ✅ Pas de trading réel sans configuration Coinbase
- ✅ Clés API sécurisées dans .env
- ✅ Logs détaillés pour debug

## 📞 Support

1. **Vérifiez les logs** : `crypto_trading_bot.log`
2. **Testez la config** : `python get_chat_id.py`
3. **Redémarrez** le bot si nécessaire

---

## 🚀 PRÊT À UTILISER !

Votre bot de trading Telegram est prêt ! 
Il vous suffit d'ajouter votre token Telegram dans le fichier `.env`.
