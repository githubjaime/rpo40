# Módulo de formulário

# Imports
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, RadioField, SubmitField, StringField, EmailField, PasswordField, BooleanField
from wtforms.validators import DataRequired, InputRequired
from datetime import datetime

# Função para o range de anos
def set_years():
    now = datetime.now().year
    years = [x for x in range(now, now - 51, -1)]
    return years

# Classe para o formulário de dados dos veículos
class inputForm(FlaskForm):
    year = SelectField("Ano", choices=[], validators=[DataRequired()])
    odometer = IntegerField("Kilometragem", validators=[DataRequired()])
    make = SelectField("Fabricante", choices=[], validators=[DataRequired()])
    model = SelectField("Modelo", choices=[], validators=[DataRequired()])
    transmission = RadioField("Transmissão", choices=['Automatic', 'Manual'], validators=[DataRequired()])
    submit = SubmitField("Enviar")

# Classe para cadastro de conta
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = EmailField('Email', validators=[DataRequired()])
    password = PasswordField('Senha', validators=[DataRequired()])
    register = SubmitField("Criar Conta")

# Classe para login
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Senha', validators=[InputRequired()])
    login = SubmitField("Login")
