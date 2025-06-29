#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour obtenir automatiquement votre Chat ID Telegram
"""

import requests
import json
import sys

def get_chat_id():
    print("🤖 RÉCUPÉRATION DE VOTRE CHAT ID TELEGRAM")
    print("=" * 50)
    
    # Demander le token
    token = input("🔑 Entrez votre Token Telegram Bot (de @BotFather) : ").strip()
    
    if not token:
        print("❌ Token requis !")
        return
    
    print("\n📱 ÉTAPES :")
    print("1. Allez sur Telegram")
    print("2. Cherchez votre bot et envoyez-lui un message")
    print("3. Revenez ici et appuyez sur Entrée")
    input("\n⏰ Appuyez sur Entrée après avoir envoyé un message à votre bot...")
    
    try:
        # Récupérer les updates
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        
        print("\n🔄 Récupération des messages...")
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Erreur HTTP {response.status_code}")
            print("Vérifiez votre token Bot")
            return
        
        data = response.json()
        
        if not data.get('ok'):
            print("❌ Erreur API Telegram")
            print(f"Message: {data.get('description', 'Erreur inconnue')}")
            return
        
        if not data.get('result'):
            print("❌ Aucun message trouvé")
            print("Assurez-vous d'avoir envoyé un message à votre bot")
            return
        
        # Extraire les Chat IDs
        chat_ids = set()
        for update in data['result']:
            if 'message' in update and 'chat' in update['message']:
                chat_id = update['message']['chat']['id']
                chat_ids.add(chat_id)
        
        if not chat_ids:
            print("❌ Aucun Chat ID trouvé")
            return
        
        print("\n✅ CHAT ID(S) TROUVÉ(S) :")
        print("-" * 30)
        
        for i, chat_id in enumerate(chat_ids, 1):
            print(f"Chat ID #{i}: {chat_id}")
        
        if len(chat_ids) == 1:
            main_chat_id = list(chat_ids)[0]
            print(f"\n🎯 VOTRE CHAT ID : {main_chat_id}")
            
            # Proposer de mettre à jour le .env
            update_env = input("\n💾 Voulez-vous mettre à jour le fichier .env ? (y/N): ")
            if update_env.lower() == 'y':
                update_env_file(token, main_chat_id)
                
                # Test d'envoi
                test_msg = input("\n🧪 Envoyer un message de test ? (y/N): ")
                if test_msg.lower() == 'y':
                    send_test_message(token, main_chat_id)
        else:
            print(f"\n⚠️  Plusieurs Chat IDs trouvés ({len(chat_ids)})")
            print("Utilisez celui qui correspond à votre conversation privée avec le bot")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur réseau : {e}")
    except Exception as e:
        print(f"❌ Erreur : {e}")

def update_env_file(token, chat_id):
    """Met à jour le fichier .env"""
    try:
        env_content = f"""# Configuration Bot de Trading Crypto
# Généré automatiquement

# TELEGRAM (OBLIGATOIRE)
TELEGRAM_BOT_TOKEN={token}
TELEGRAM_CHAT_ID={chat_id}

# PARAMÈTRES DU BOT
TRADING_MODE=SIMULATION
PREDICTION_THRESHOLD=0.00002
VOLUME_GROWTH_FACTOR=2.0
PRICE_SURGE_PERCENTAGE=0.01

# COINBASE PRO (OPTIONNEL)
COINBASE_API_KEY=
COINBASE_API_SECRET=
COINBASE_PASSPHRASE=

# EMAIL (OPTIONNEL)
EMAIL_SENDER=
EMAIL_PASSWORD=
EMAIL_RECIPIENT=
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
"""
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("✅ Fichier .env mis à jour !")
        
    except Exception as e:
        print(f"❌ Erreur mise à jour .env : {e}")

def send_test_message(token, chat_id):
    """Envoie un message de test"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        message = """🎉 Configuration réussie !

🤖 Votre bot de trading est prêt !
📊 Vous recevrez les alertes ici.

✅ Chat ID confirmé
📱 Test de communication réussi

🚀 Prochaines étapes :
1. Lancez : python coinbase_telegram_bot.py
2. Surveillez les alertes de trading !"""
        
        data = {
            'chat_id': chat_id,
            'text': message
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ Message de test envoyé avec succès !")
            print("📱 Vérifiez votre Telegram")
        else:
            print(f"❌ Erreur envoi message : {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erreur test : {e}")

def main():
    try:
        get_chat_id()
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir !")
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")

if __name__ == "__main__":
    main()
