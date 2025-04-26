import os

from flask import Flask, send_file

app = Flask(__name__)

@app.route("/")
def index():
    return send_file('src/login.html')

def main():
    # Change the port to 5000 or another non-privileged port
    app.run(port=int(os.environ.get('PORT', 5000)))

if __name__ == "__main__":
    main()