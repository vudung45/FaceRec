import pyrebase




config = {
  "apiKey": "AIzaSyAK1iztrB4nrYXQaw6xBgi2D1t4ID9gfYY",
  "authDomain": "facialrecognition-b1916.firebaseapp.com",
  "databaseURL": "https://facialrecognition-b1916.firebaseio.com",
  "storageBucket": "facialrecognition-b1916.appspot.com"
}

firebase = pyrebase.initialize_app(config)

# Get a reference to the auth service
auth = firebase.auth()



email = "robertcook.rdc@gmail.com"
password = ""

#auth.create_user_with_email_and_password(email, password)

# Log the user in
user = auth.sign_in_with_email_and_password("robertcook.rdc@gmail.com", "General2012")

db = firebase.database()
