import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_channel_id():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN manquant dans .env")
        return
    
    print("🔍 Récupération de l'ID du canal...")
    print("📝 Assurez-vous que votre bot est administrateur du canal")
    print("💬 Envoyez d'abord un message dans le canal, puis appuyez sur Entrée")
    input("⏰ Appuyez sur Entrée après avoir envoyé un message dans le canal...")
    
    try:
        # Récupérer les updates
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['ok'] and data['result']:
                print("\n📋 IDs trouvés :")
                print("-" * 40)
                
                channel_ids = set()
                for update in data['result']:
                    if 'channel_post' in update:
                        channel_id = update['channel_post']['chat']['id']
                        channel_title = update['channel_post']['chat'].get('title', 'Canal sans nom')
                        channel_ids.add((channel_id, channel_title))
                        print(f"📢 Canal: {channel_title}")
                        print(f"🆔 ID: {channel_id}")
                        print("-" * 40)
                
                if channel_ids:
                    # Prendre le premier canal trouvé
                    channel_id, channel_title = list(channel_ids)[0]
                    print(f"\n✅ ID du canal à utiliser : {channel_id}")
                    
                    # Test d'envoi
                    test_message(token, channel_id, channel_title)
                    
                    return channel_id
                else:
                    print("❌ Aucun canal trouvé")
                    print("Assurez-vous que :")
                    print("1. Le bot est administrateur du canal")
                    print("2. Vous avez envoyé un message dans le canal")
            else:
                print(f"❌ Erreur API : {data}")
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Erreur : {e}")

def test_message(token, channel_id, channel_title):
    """Test d'envoi de message dans le canal"""
    test = input(f"\n🧪 Tester l'envoi dans '{channel_title}' ? (y/N): ")
    
    if test.lower() == 'y':
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            message = f"🤖 Test du bot de trading\n📊 Canal configuré avec succès !\n⏰ Test envoyé à {requests.get('http://worldtimeapi.org/api/timezone/Europe/Paris').json()['datetime'][:19]}"
            
            data = {
                'chat_id': channel_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                print("✅ Message de test envoyé avec succès !")
                print("📱 Vérifiez votre canal Telegram")
                
                # Proposer de mettre à jour .env
                update_env = input("\n💾 Mettre à jour le fichier .env ? (y/N): ")
                if update_env.lower() == 'y':
                    update_env_file(channel_id)
            else:
                print(f"❌ Erreur envoi : {response.json()}")
                
        except Exception as e:
            print(f"❌ Erreur test : {e}")

def update_env_file(channel_id):
    """Met à jour le TELEGRAM_CHAT_ID dans .env"""
    try:
        # Lire le fichier .env existant
        env_content = ""
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                env_content = f.read()
        
        # Remplacer ou ajouter TELEGRAM_CHAT_ID
        lines = env_content.split('\n')
        updated = False
        
        for i, line in enumerate(lines):
            if line.startswith('TELEGRAM_CHAT_ID='):
                lines[i] = f'TELEGRAM_CHAT_ID={channel_id}'
                updated = True
                break
        
        if not updated:
            lines.append(f'TELEGRAM_CHAT_ID={channel_id}')
        
        # Sauvegarder
        with open('.env', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ Fichier .env mis à jour !")
        print(f"📢 TELEGRAM_CHAT_ID = {channel_id}")
        
    except Exception as e:
        print(f"❌ Erreur mise à jour .env : {e}")

if __name__ == "__main__":
    get_channel_id()