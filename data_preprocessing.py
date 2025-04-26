import sqlite3

DATABASE = 'users.db'

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT
            )
        ''')
        db.commit()

init_db()

def add_user(username, password):
    """Adds a user to the database."""
    with get_db() as db:
        try:
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            db.commit()
        except sqlite3.IntegrityError as e:
            raise sqlite3.IntegrityError("User already exists.") from e

def check_user(username, password):
    """Checks if a user exists in the database with the given password."""
    with get_db() as db:
        cursor = db.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        if user:
            return user['password'] == password
        else:
            return False