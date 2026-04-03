import os
import pymongo
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize MongoDB connection.
        """
        self.uri = os.getenv("MONGO_URI")
        if not self.uri:
            print("Warning: MONGO_URI not found in environment variables.")
            self.db = None
            self.users_collection = None
            return

        try:
            # Create a new client and connect to the server
            self.client = pymongo.MongoClient(self.uri, server_api=ServerApi('1'))
            
            # Send a ping to confirm a successful connection
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            
            self.db = self.client.get_database("fashion_app")
            self.users_collection = self.db.get_collection("users")
            
        except Exception as e:
            print(f"MongoDB Connection Error: {e}")
            self.db = None
            self.users_collection = None

    def get_user(self, username):
        """
        Retrieve user document by username.
        """
        if self.users_collection is None:
            return None
        return self.users_collection.find_one({"username": username})

    def create_user(self, username, password):
        """
        Create a new user. Returns True if successful, False if user already exists.
        """
        if self.users_collection is None:
            return False
            
        if self.get_user(username):
            return False
            
        user_doc = {
            "username": username,
            "password": password,
            "saved_looks": [],
            "created_at": str(datetime.now())
        }
        self.users_collection.insert_one(user_doc)
        return True

    def add_saved_look(self, username, look_data):
        """
        Add a saved look to the user's history.
        """
        if self.users_collection is None:
            return False
            
        self.users_collection.update_one(
            {"username": username},
            {"$push": {"saved_looks": look_data}}
        )
        return True

    def verify_password(self, username, password):
        """
        Verify credentials.
        """
        user = self.get_user(username)
        if user and user['password'] == password:
            return True
        return False

# Quick test if run directly
from datetime import datetime
if __name__ == "__main__":
    db = Database()
    if db.db is not None:
        print("Test Connection Success")
