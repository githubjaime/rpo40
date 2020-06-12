# Hangman Game (Jogo da Forca) 
# Programação Orientada a Objetos

# Import
import random

# Board (tabuleiro)
board = ['''

>>>>>>>>>>Hangman<<<<<<<<<<

+---+
|   |
    |
    |
    |
    |
=========''', '''

+---+
|   |
O   |
    |
    |
    |
=========''', '''

+---+
|   |
O   |
|   |
    |
    |
=========''', '''

 +---+
 |   |
 O   |
/|   |
     |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
     |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
/    |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
/ \  |
     |
=========''']


# Classe
class Hangman:

	# Método Construtor
	def __init__(self, word):
		self.word = word
		self.missed_letters = []
		self.guessed_letters = []
		
	# Método para adivinhar a letra
	def guess(self, letter):
		if letter in self.word and letter not in self.guessed_letters:
			self.guessed_letters.append(letter)
		elif letter not in self.word and letter not in self.missed_letters:
			self.missed_letters.append(letter)
		else:
			return False
		return True
		
	# Método para verificar se o jogo terminou
	def hangman_over(self):
		return self.hangman_won() or (len(self.missed_letters) == 6)
		
	# Método para verificar se o jogador venceu
	def hangman_won(self):
		if '_' not in self.hide_word():-Avaliação da proposta de pesquisa, pela Comissão de Avaliação designada pela CCP, em relação à clareza do texto, objetivos, justificativa, procedimentos e se o tema proposto se enquadra nos requisitos necessários para o doutorado. Considerando a qualidade geral do plano será atribuída nota de 1 a 5 (1- Muito fraco, 2- Fraco, 3- Razoável, 4- Bom, 5- Muito bom); 2-Avaliação do currículo e histórico escolar, pela Comissão de Avaliação designada pela CCP, de acordo com os seguintes critérios: • Iniciação científica: 0,50 pontos por ano; • Especialização (360 horas): 1,0 ponto na área ou correlata e 0,5 ponto em outra área; • Artigo completo em anais de congresso nacional: 0,25 pontos por artigo, limitado a 3 artigos; • Artigo completo em anais de congresso internacional: 0,5 ponto por artigo, limitado a 3 artigos; • Artigo completo em periódico não indexado: 0,75 pontos por artigo; • Artigo completo em periódico indexado com JCR ou SJR: 1,5 pontos por artigo; • Livro: 1,0 pontos por livro como autor, coautor, coordenador, organizador ou editor; • Capítulo de livro: 0,5 ponto por capítulo, limitado a 3; • Patente de produto ou processo: 0,5 pontos por patente e 2,0 pontos por patente licenciada; • Software desenvolvido e registrado: 0,5 ponto por produto. Atenção: As 2 (duas) notas (Proposta de Pesquisa e Avaliação do Currículo/Histórico Escolar) serão somadas e, a partir do resultado, será feita a classificação dos candidatos em ordem decrescente, considerando-se para aprovação obter 4 (quatro) pontos como pontuação mínima e disponibilidade de vaga do orie
			return True
		return False
		
	# Método para não mostrar a letra no board
	def hide_word(self):
		rtn = ''
		for letter in self.word:
			if letter not in self.guessed_letters:
				rtn += '_'
			else:
				rtn += letter
		return rtn
		
	# Método para checar o status do game e imprimir o board na tela
	def print_game_status(self):
		print (board[len(self.missed_letters)])
		print ('\nPalavra: ' + self.hide_word())
		print ('\nLetras erradas: ',) 
		for letter in self.missed_letters:
			print (letter,) 
		print ()
		print ('Letras corretas: ',)
		for letter in self.guessed_letters:
			print (letter,)
		print ()

# Método para ler uma palavra de forma aleatória do banco de palavras
def rand_word():
        with open("palavras.txt", "rt") as f:
                bank = f.readlines()
        return bank[random.randint(0,len(bank))].strip()

# Método Main - Execução do Programa
def main():

	# Objeto
	game = Hangman(rand_word())

	# Enquanto o jogo não tiver terminado, print do status, solicita uma letra e faz a leitura do caracter
	while not game.hangman_over():
		game.print_game_status()
		user_input = input('\nDigite uma letra: ')
		game.guess(user_input)

	# Verifica o status do jogo
	game.print_game_status()	

	# De acordo com o status, imprime mensagem na tela para o usuário
	if game.hangman_won():
		print ('\nParabéns! Você venceu!!')
	else:
		print ('\nGame over! Você perdeu.')
		print ('A palavra era ' + game.word)
		
	print ('\nFoi bom jogar com você! Agora vá estudar!\n')

# Executa o programa		
if __name__ == "__main__":
	main()
