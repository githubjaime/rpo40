print("-"*30)
print("sequencia de Fibonacci")
print("-"*30)
n = int(input("quantos termos?"))
t1 = 0
t2 = 1
print("{} -> {}".format(t1, t2), end="")
cont = 3
while cont <= n:
  t3 = t1 + t2
  print(" -> {}".format(t3), end="")
  cont += 1
  t1 = t2
  t2 = t3
print("-> Fim")

      
        
