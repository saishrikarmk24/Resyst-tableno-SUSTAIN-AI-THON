from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user

from . import db

auth = Blueprint('auth', __name__)

@auth.route('/sign-up', methods = ['GET', 'POST'])
def signup():

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')


        print(f"Checking username: {username}")
        user = User.query.filter_by(username=username).first()
        if user: 
            flash('Username already Exists', category='Error')
        elif len(email) < 5:
            flash('Email must be greater han 4 characters.', category='Error')
        elif len(username) < 2:
            flash('Username must be greater than 1 character.', category='Error')
        elif password != confirm_password:
            flash("Passwords don't match.", category="Error")
        elif len(password) < 7:
            flash("Password must be at least 7 characters.", category="Error")
        else:
            new_user = User(email=email, username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()

            user = User.query.filter_by(email=email).first()
            if user is None:
                return jsonify({'error': 'User not found!'})
            
            login_user(user, remember=True)
            flash("Account Created!", category="Success")
            print(new_user)
            print(email)
            return redirect(url_for('views.home'))

    return render_template('sign-up.html')

@auth.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user:
            print(f"User found in database - Username: '{user.username}'")
            if check_password_hash(user.password, password):
                print("Password match successful")
                login_user(user, remember=True)
                flash('Logged In Successfully!', category='Success')
                return redirect(url_for('views.home'))
            else:
                print("Password mismatch")
                flash('Incorrect Password', category='Error')
        else:
            print("No matching user found")
            flash("Username doesn't Exist", category='Error')
    
    return render_template('login.html', boolean = False) 


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
'''

@auth.route('/debug-users')
def debug_users():
    users = User.query.all()
    for user in users:
        print(f"User: {user.username}, Email: {user.email}")
    return "Check console for user list."

@auth.route('/test-query')
def test_query():
    user = User.query.filter_by(username=username).first()
    if user:
        return f"User found: {user.username}, {user.email}"
    return "User not found"

'''


@auth.route('/analyse')
def analyse():
    return render_template(url_for('auth.analyse'))