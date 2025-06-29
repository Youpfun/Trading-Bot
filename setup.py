import os
import sys

def setup_bot():
    """Configuration automatique du bot de trading"""
    print("🤖 Configuration du Bot de Trading Crypto")
    print("="*50)
    
    # Vérifier si .env existe déjà
    if os.path.exists('.env'):
        response = input("📁 Le fichier .env existe déjà. Le remplacer? (y/N): ")
        if response.lower() != 'y':
            print("❌ Configuration annulée")
            return
    
    print("\n📱 CONFIGURATION TELEGRAM")
    print("-" * 30)
    print("1. Créez un bot avec @BotFather sur Telegram")
    print("2. Envoyez /newbot et suivez les instructions")
    print("3. Copiez le token fourni")
    
    telegram_token = input("\n🔑 Token Telegram Bot: ").strip()
    if not telegram_token:
        print("❌ Token Telegram requis!")
        return
    
    print("\n4. Envoyez un message à votre bot")
    print("5. Allez sur: https://api.telegram.org/bot{}/getUpdates".format(telegram_token))
    print("6. Trouvez votre chat_id dans la réponse")
    
    chat_id = input("\n🆔 Votre Chat ID: ").strip()
    if not chat_id:
        print("❌ Chat ID requis!")
        return
    
    print("\n⚙️ PARAMÈTRES DU BOT")
    print("-" * 25)
    
    # Paramètres avec valeurs par défaut
    trading_mode = input("🤖 Mode trading (SIMULATION/REAL) [SIMULATION]: ").strip() or "SIMULATION"
    threshold = input("📊 Seuil de prédiction [0.00002]: ").strip() or "0.00002"
    volume_factor = input("📈 Facteur volume [2.0]: ").strip() or "2.0"
    price_surge = input("💹 Seuil hausse prix [0.01]: ").strip() or "0.01"
    
    # Configuration Coinbase (optionnelle)
    print("\n💰 COINBASE PRO (Optionnel - pour trading réel)")
    print("-" * 45)
    print("⚠️  Laissez vide si vous voulez seulement les alertes")
    
    coinbase_key = input("🔑 Coinbase API Key (optionnel): ").strip()
    coinbase_secret = input("🔐 Coinbase Secret (optionnel): ").strip()
    coinbase_passphrase = input("🎫 Coinbase Passphrase (optionnel): ").strip()
    
    # Création du fichier .env
    env_content = f"""# Configuration Bot de Trading Crypto
# Généré automatiquement le {os.path.basename(__file__)}

# TELEGRAM (OBLIGATOIRE)
TELEGRAM_BOT_TOKEN={telegram_token}
TELEGRAM_CHAT_ID={chat_id}

# PARAMÈTRES DU BOT
TRADING_MODE={trading_mode}
PREDICTION_THRESHOLD={threshold}
VOLUME_GROWTH_FACTOR={volume_factor}
PRICE_SURGE_PERCENTAGE={price_surge}

# COINBASE PRO (OPTIONNEL)
COINBASE_API_KEY={coinbase_key}
COINBASE_API_SECRET={coinbase_secret}
COINBASE_PASSPHRASE={coinbase_passphrase}

# EMAIL (OPTIONNEL - Legacy)
EMAIL_SENDER=
EMAIL_PASSWORD=
EMAIL_RECIPIENT=
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587

# CRYPTO.COM (OPTIONNEL - Legacy)
CRYPTO_COM_API_KEY=
CRYPTO_COM_API_SECRET=
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("\n✅ CONFIGURATION TERMINÉE!")
        print("="*50)
        print("📁 Fichier .env créé avec succès")
        print("\n🚀 PROCHAINES ÉTAPES:")
        print("1. Installez les dépendances: pip install -r requirements.txt")
        print("2. Lancez le bot: python coinbase_telegram_bot.py")
        print("3. Vous recevrez les alertes sur Telegram!")
        
        print("\n📋 RÉSUMÉ DE LA CONFIGURATION:")
        print(f"🤖 Bot Token: {telegram_token[:10]}...")
        print(f"💬 Chat ID: {chat_id}")
        print(f"⚙️  Mode: {trading_mode}")
        print(f"📊 Seuil: {threshold}")
        
        if coinbase_key:
            print("💰 Coinbase: Configuré")
        else:
            print("💰 Coinbase: Mode alerte uniquement")
            
    except Exception as e:
        print(f"❌ Erreur lors de la création du fichier .env: {e}")
        return
    
    # Test de la configuration
    test_config = input("\n🧪 Tester la configuration Telegram? (y/N): ")
    if test_config.lower() == 'y':
        test_telegram_config(telegram_token, chat_id)

def test_telegram_config(token, chat_id):
    """Test la configuration Telegram"""
    try:
        print("\n🧪 Test de la configuration Telegram...")
        
        import requests
        
        # Test API Telegram
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info['ok']:
                print(f"✅ Bot trouvé: {bot_info['result']['first_name']}")
                
                # Test envoi message
                message_url = f"https://api.telegram.org/bot{token}/sendMessage"
                test_message = {
                    'chat_id': chat_id,
                    'text': '🎉 Configuration réussie!\n\n🤖 Votre bot de trading est prêt!\n📊 Vous recevrez les alertes ici.',
                    'parse_mode': 'HTML'
                }
                
                msg_response = requests.post(message_url, json=test_message, timeout=10)
                
                if msg_response.status_code == 200:
                    print("✅ Message de test envoyé avec succès!")
                    print("📱 Vérifiez votre Telegram")
                else:
                    print(f"❌ Erreur envoi message: {msg_response.text}")
            else:
                print(f"❌ Erreur API: {bot_info}")
        else:
            print(f"❌ Erreur HTTP: {response.status_code}")
            
    except ImportError:
        print("⚠️  Module 'requests' manquant. Installez d'abord les dépendances:")
        print("pip install requests")
    except Exception as e:
        print(f"❌ Erreur test: {e}")

def install_dependencies():
    """Installe les dépendances automatiquement"""
    print("📦 Installation des dépendances...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dépendances installées avec succès!")
            return True
        else:
            print(f"❌ Erreur installation: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("🚀 SETUP BOT DE TRADING CRYPTO")
    print("=" * 50)
    
    # Vérifier requirements.txt
    if not os.path.exists('requirements.txt'):
        print("❌ Fichier requirements.txt manquant!")
        return
    
    # Proposer installation des dépendances
    install_deps = input("📦 Installer les dépendances maintenant? (Y/n): ")
    if install_deps.lower() != 'n':
        if not install_dependencies():
            print("⚠️  Continuez avec la configuration, installez manuellement plus tard")
    
    # Configuration
    setup_bot()
    
    print("\n🎯 CONFIGURATION TERMINÉE!")
    print("Consultez le README.md pour plus d'informations")

if __name__ == "__main__":
    main()
