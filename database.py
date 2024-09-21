from pymongo import MongoClient
import bcrypt
# from bson import ObjectId

class Database():
    baseAddress = '/home/jwu283/shared/Fingerprint_demo'
    def __init__(self) -> None:
        self.client = MongoClient("[REDACTED]")
        self.db = self.client['fingerprints']
        self.collection = self.db['users']

    def check_valid_username(self, username):
        user = self.collection.find_one({"username": username})
        if user is None:
            return True
        return False

    def add_user(self, username, password, path1, path2):
        salt = bcrypt.gensalt()
        p_bytes = password.encode('utf-8')
        hashed_password = bcrypt.hashpw(p_bytes, salt)
        post = {"username": username, "password": hashed_password,
                "leftFingerprintPath": path1, "rightFingerprintPath": path2}
        self.collection.insert_one(post)
        return True
    
    def get_user(self, username, password):
        stored_user = self.collection.find_one({"username": username})
        stored_hashed_password = stored_user["password"]
        
        p_bytes = password.encode('utf-8')

        # Check if the provided password matches the stored hashed password
        if bcrypt.checkpw(p_bytes, stored_hashed_password):
            return stored_user
        else:
            return None
    
    def get_all_users(self):
        users = self.collection.find({})
        return users
    
    def delete_user(self, username):
        self.collection.delete_one({"username": username})
        return True

if __name__ == '__main__':
    db = Database()
    db.add_user("name", "password", "path1", "path2")
