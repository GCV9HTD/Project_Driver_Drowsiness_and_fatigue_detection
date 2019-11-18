
import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "smartcaralert237@gmail.com"  # Enter your address
receiver_email = "tsopnangsr@gmail.com"  # Enter receiver address
password = "SmartCarAlert237"
message = """From: From Smart Car 237 <smartcaralert237@gmail.com>
To: To Romaric Tsopnang <tsopnangsr@gmail.com>
Subject: Drowsiness Detected

The driver in the car imatriculated LT125OU is Drowsy.
"""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)
