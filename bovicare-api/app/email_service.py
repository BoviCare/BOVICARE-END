import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app
import os

class EmailService:
    def __init__(self):
        # Configurações de email (para desenvolvimento local)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        
    def send_password_reset_email(self, to_email, code, username):
        """Envia email com código de recuperação de senha"""
        try:
            # Verificar se temos configurações de e-mail
            if not self.email_user or not self.email_password:
                print(f"📧 EMAIL SIMULADO - Para: {to_email}")
                print(f"📧 Código: {code}")
                print(f"📧 Usuário: {username}")
                print(f"📧 Assunto: Recuperação de senha - BoviCare")
                print(f"📧 Mensagem: Olá {username}, seu código de recuperação é: {code}")
                print("=" * 50)
                print("⚠️  Configure as variáveis de ambiente para envio real:")
                print("   EMAIL_USER=seu-email@gmail.com")
                print("   EMAIL_PASSWORD=sua-senha-de-app")
                return True
            
            # Envio real de e-mail
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = to_email
            msg['Subject'] = "Recuperação de senha - BoviCare"
            
            body = f"""
            Olá {username},
            
            Você solicitou a recuperação de senha para sua conta no BoviCare.
            
            Seu código de verificação é: {code}
            
            Este código expira em 30 minutos.
            
            Se você não solicitou esta recuperação, ignore este e-mail.
            
            Atenciosamente,
            Equipe BoviCare
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, to_email, text)
            server.quit()
            
            print(f"📧 EMAIL ENVIADO - Para: {to_email}")
            print(f"📧 Código: {code}")
            return True
            
        except Exception as e:
            print(f"Erro ao enviar email: {str(e)}")
            return False

class SMSService:
    def __init__(self):
        # Para desenvolvimento, vamos simular o envio de SMS
        self.api_key = os.getenv('SMS_API_KEY', '')
        self.api_url = os.getenv('SMS_API_URL', '')
        
    def send_password_reset_sms(self, to_phone, code, username):
        """Envia SMS com código de recuperação de senha"""
        try:
            # Para desenvolvimento, vamos apenas simular o envio
            print(f"📱 SMS SIMULADO - Para: {to_phone}")
            print(f"📱 Código: {code}")
            print(f"📱 Usuário: {username}")
            print(f"📱 Mensagem: BoviCare - Seu código de recuperação é: {code}")
            print("=" * 50)
            
            # Em produção, implemente aqui a integração com provedor de SMS
            # Exemplo com Twilio, AWS SNS, etc.
            
            return True
        except Exception as e:
            print(f"Erro ao enviar SMS: {str(e)}")
            return False

# Instâncias globais dos serviços
email_service = EmailService()
sms_service = SMSService()
