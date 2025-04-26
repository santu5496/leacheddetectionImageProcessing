import sqlite3
from sqlite3 import OperationalError
from flask import Flask, request, jsonify


app = Flask(__name__)

# Database setup
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

def add_user(username, password):
    with get_db() as db:
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        db.commit()
def check_user(username, password):
    with get_db() as db:
        cursor = db.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        return cursor.fetchone() is not None
        ''')
        db.commit()

init_db()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    if check_user(username, password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data['username']
    password = data['password']
    try:
        add_user(username, password)
        return jsonify({'message': 'User registered successfully'}), 201
    except ValueError as e:
        return jsonify({'message': str(e)}), 400
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Username already exists'}), 400
    except OperationalError:
        return jsonify({'message': 'Database is locked'}), 400



