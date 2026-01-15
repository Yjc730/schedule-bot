import urllib.parse
import webbrowser

def send_email_via_outlook(recipient_email: str, body: str):
    subject = "通知"

    url = (
        "https://outlook.office.com/mail/deeplink/compose?"
        f"to={urllib.parse.quote(recipient_email)}"
        f"&subject={urllib.parse.quote(subject)}"
        f"&body={urllib.parse.quote(body)}"
    )

    print("📧 Opening Outlook compose page")
    webbrowser.open(url)
