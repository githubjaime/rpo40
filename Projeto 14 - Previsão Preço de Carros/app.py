# Módulo principal da app

# Imports
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_login import UserMixin

# Cria a app Flask
app = Flask(__name__)

# Credenciais para o banco de dados de usuários
app.config['SECRET_KEY'] = 'anything_will_be_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Instância do database e inicialização
db = SQLAlchemy(app)

# Bootstrap para dimensionamento automático da página
Bootstrap(app)

# Navegação pelos menus
nav = Nav()

# Inicialização da navegação
nav.init_app(app)

# Classe para dados do usuário
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(25))
    password = db.Column(db.String(80))

# Barra de navegação
@nav.navigation()
def mynavbar():
    return Navbar('Machine Learning Web App',
                  View('Previsões', 'predict'),
                  View('Relatórios', 'index'),
                  View('Sair', 'logout'))

# Módulo main
if __name__ == '__main__':
    app.run()

# Import das rotas para as páginas web
import routes.routes

