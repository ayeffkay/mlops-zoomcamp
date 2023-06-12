import os
from prefect_email.credentials import EmailServerCredentials


def create_email_server_credentials():
    credentials = EmailServerCredentials(
        username=os.getenv("user_name"),
        password=os.getenv("user_password"),
    )
    credentials.save("email-credentials", overwrite=True)


if __name__ == "__main__":
    create_email_server_credentials()
