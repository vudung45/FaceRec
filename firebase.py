import pyrebase




config = {
  "apiKey": "",
  "authDomain": "",
  "databaseURL": "",
  "storageBucket": ""
}

firebase = pyrebase.initialize_app(config)

# Get a reference to the auth service
auth = firebase.auth()



email = ""
password = ""

#auth.create_user_with_email_and_password(email, password)

# Log the user in
user = auth.sign_in_with_email_and_password("", "")

db = firebase.database()
