'''
    Calcul Numeric
    Tema 1
    Badea Adrian Catalin, grupa 334, anul III, FMI
'''


import numpy as np
import matplotlib.pyplot as plt
# Vom folosi functiile din matplotlib pentru graficele functiilor
# Fiecare plot va fi salvat atat in folder-ul in care ruleaza sursa, cat si in IDE (daca acesta are panou de "Plots")


'''
Exericitiul 1.

Enunt:
    
Sa se gaseasca o aprixmare cu 7 zecimale a numarului sqrt(7)

Rezolvare:
    
    Vom cauta folosing metoda Newton Raphson pe sirul [a, b], 
    pentru functia  (x ^ 2) - 7
    
    Vom alege a = 0, b = 7, deoarece
    sqrt(x) >= 0, pentru orice x >= 0
    sqrt(x) <= x, pentru orice x >= 1
    
    Deoarece f''(x0) = ((x ^ 2) - 7)'' = 2 >= 0, f este convexa
    si putem alege x0 = 3
    pentru x0 = 3, f(x0) * f''(x0) = 2 * 2 = 4 > 0
'''

def fun(x):
    #Functia f
    return x ** 2  - 7

def fun_derivative(x):
    #Derivata functiei f, f'
    return 2 * x 
 
# Parametrii metodei newton raphson
def newton_raphson(x0, fun, fun_derivative, eps):
    
    last_x = x0
    iter_num = 0
    
    while True:
        
        iter_num += 1
        new_x = last_x - fun(last_x) / fun_derivative(last_x)
        last_x = new_x
        if abs(fun(last_x)) < eps: 
           break
       
    return last_x, iter_num


# Datele noastre de intrare conform sectiunii "Rezolvare"
    
EPS = 10**(-7)
x0 = 3
a = 0
b = 7
x_num, iter_num = newton_raphson(x0, fun, fun_derivative, EPS)


print('Metoda Newton Raphson')
print('Ecuatia: x^2 - 7')
print('Intervalul: [', a, ',', b, ']')
print('Solutia numerica: x_num = ', x_num)
print('Numarul de iteratii: iter_num = {}'.format(iter_num))
print('-' * 50)


'''
Exercitiul 2.

Enunt.
    Rezolvati numeric ecuatia e ^ (x - 2) = cos(e ^ (x - 2)) + 1. Sa se ilustreze
    grafic cele doua functii si punctul lor de intersectie.
    
Rezolvare.
    Fie f1 functia f1(x) = e ^ (x - 2) si f2 = cos(e ^ (x - 2)) + 1
    f2(x) = cos(f1(X)) + 1
    
    Vom cauta solutiile aproximarile solutiilor ecuatiei folosing metoda bisectiei
    pentru functia f = f1 - f2
    
    -1 <= cos(e ^ (x - 2)) <= 1
    0 <= cos(e ^ (x - 2)) <= 2
    
    Din enunt avem, e ^ (x - 2) = cos(e ^ (x - 2)) + 1.
    Deci 0 <= e ^ (x - 2) <= 2
    -inf <= (x - 2)  <= ln(2)
    x <= 2 + ln(2) < 3
    
    f(2) = 1 - cos(1) - 1 = 0 - cos(1) < 0
    f(3) = e - cos(e) - 1 = (e - 1) - cos(e) > 0 
    Deci f(2) * f(3) < 0
    
    Putem alege capetele intervalului 2 si 3
    
    
'''

# Definitia functiei f1
def fun1(x):
    return np.exp(x - 2)

# Definitia functiei f2
def fun2(x) :
    return np.cos(fun1(x)) + 1
    
# Definitia functiei f
def fun(x):
    return fun1(x) - fun2(x)

# Definitia metodei bisectiei

def bisection_method(a, b, fun, eps):
    
    # Calculam numarul maxim de iteratii in functie de lungimea intervalului si de epsilon
    num_iter = int(np.floor(np.log2((b - a) / eps))) 
    
    x_num = (a + b) / 2  
    for num_iter in range(1, num_iter):
        
        # Daca f(x_current) < eps, atunci putem considera ca este 0, si am gasit solutie(avand in vedere eroarea)
        if abs(fun(x_num)) < eps:
            break
        
        # Calculam semnul functiei la mijlocul
        elif np.sign(fun(a)) * np.sign(fun(x_num)) < 0:
            b = x_num
        else:
            a = x_num
 
        x_num = (a + b) / 2

    return x_num, num_iter

# Datele conform sectiunii "Rezolvare"
a = 2
b = 3
EPS = 10**(-7)
x_num, iter_num = bisection_method(a, b, fun, EPS)

print('Metoda Bisectiei')
print('Ecuatia: e ^ (x - 2) - cos(e ^ (x - 2)) - 1')
print('Intervalul: [{:.2f},{:.2f}]'.format(a, b))
print('Solutia numerica: x_num = {:.5f}'.format(x_num))
print('Numarul de iteratii: iter_num = {}'.format(iter_num))
print('-' * 50)


x = np.linspace(a, b, 100)
y1 = fun1(x)
y2 = fun2(x)
y3 = fun(x)
plt.figure(figsize = (5, 5))
plt.plot(x, y1, linestyle='-', linewidth=3)
plt.plot(x, y2, linestyle='-', linewidth=3)
plt.plot(x, y3, linestyle='-', linewidth=3)



plt.scatter(x_num, fun1(x_num), s=100, c='black', marker='o')  
plt.scatter(x_num, fun(x_num), s=100, c='black', marker='o') 

plt.legend(['f1(x)', 'f2(x)', 'f(x) = f1(x) - f2(x)']) 
plt.xlabel('x')
plt.title('Metoda Bisectiei')
plt.grid(b=True)

plt.savefig('metoda_bisectiei.png')
plt.show()


'''
Exercitiul 3
Enunt.

    Implementati metoda pozitiei false, care aproximeaza solutiile ecuatiilor neliniare
    Sa se construiasca graficul functiei 
    f(x) = x ^ 3 + 6 * x ^ 2 + 11 * x  + 6, definita pe [-5, 5]
    Alegeti trei subinteravel astfel incat metoda pozitiei false sa fie convergenta

    
Rezolvare.    
    Functia noastra este : 
    f'(x) = 3 * x ^ 2 + 12 * x + 11
    
    Se anuleaza in -2.57, -1.42 (cu aproximare)
    
    f''(x) = 6 * x + 12
    Se anuleaza in x = -2
                                   
    
    a = [-5, -2.1, -1.3]
    b = [-2.6, -1.8, 5]


'''


# Functia noastra polinomiala
def fun(x):
    return (x ** 3) + 6 * (x ** 2) + 11 * x + 6

# Definim functia care ne ajuta in metoda pozitiei false
    
def fun_helper(a, b):
    return (a * fun(b) - b * fun(a)) / (fun(b) - fun(a))

def pozitie_falsa(a, b, fun, fun_helper, eps):
    
    last_x = fun_helper(a, b)
    iter_num = 0
    assert fun(a) * fun(b) < 0
    
    while True:
        
        if (abs(fun(last_x)) < eps):
            return last_x, iter_num
        
        elif fun(a) * fun(last_x) < 0 :
            b = last_x  
            new_x = fun_helper(a, b)
            
        elif fun(a) * fun(last_x) > 0 :
            a = last_x
            new_x = fun_helper(a, b)
            
        iter_num += 1
        
        
        if (abs(new_x - last_x)  < abs(eps * last_x)):
            break
        
        last_x = new_x
        
        
    return last_x, iter_num


EPS = 10**(-7)
a = [-5, -2.1, -1.3]
b = [-2.6, -1.8, 5]

print('Metoda Pozitiei false')
print('Ecuatia: x ^ 3 + 6 * x ^ 2 + 11 * x  + 6')

for i in range(len(a)):    
    print('Intervalele: [{:.2f},{:.2f}]'.format(a[i], b[i]))
    
    
    x_num, iter_num = pozitie_falsa(a[i], b[i], fun, fun_helper, EPS)
    
    print('Solutia numerica: x_num = {:.5f}'.format(x_num))

    print('Numarul de iteratii: iter_num = {}'.format(iter_num))
    
    print('-' * 50)
    plt.scatter(x_num, fun(x_num), s=100, c='black', marker='o')  


    
x = np.linspace(-5, 5 , 100)
y = fun(x)
plt.plot(x, y, linestyle='-', linewidth=3)

plt.legend(['f(x)']) 
plt.xlabel('x')
plt.title('Metoda Pozitiei False')
plt.grid(b = True)


plt.savefig('metoda_pozitiei_false.png')
plt.show()

'''
Exercitiul 4
Enunt.

    Implementati metoda secantei
    Intr-un fisier sa se construiasca graficul functiei 
    f(x) = x ^ 3 - x ^ 2 - 4 * x  + 3, definita pe [-3, 3]
    Alegeti trei subinteravle astfel incat metoda secantei sa fie convergenta
    Aflati cele trei solutii apeland functia secanta cu eroarea de aproximare eps = 10−5


Rezolvare.

    Observam ca pentru diferite perechi de valori ale functiei f, precum 
    f(-3) * f(-1.5) < 0
    f(-1) * f(0.8) < 0
    f(2) * f(3)
    Alegem astfel capetele intervalelor, acestea fiind si in vecinitatile solutiilor reale

'''

def fun(x):
    return (x ** 3) -  (x ** 2) - 4 * x + 3

def metoda_secantei(fun, a, b, x0, x1, eps):
       
    iter_num = 0
    while (abs(x1 - x0) > abs(eps * x0)):
        
        # Verificam ca numitorul fractiei urmatoare este mai mare decat 0 in modul            
        assert abs(fun(x1) - fun(x0)) > eps, 'Valori gresite pentru x0, x1 pentru impartire'
        
        x_aux = (x0 * fun(x1) - x1 * fun(x0)) / (fun(x1) - fun(x0))
        
        # Dupa iteratia curenta, x ajunge in afara intervalului, deci trebuie alese alte valori de intrare
        
        assert a <= x_aux and x_aux <= b, 'Valori gresite pentru x0, x1, iesire din interval'
        
        iter_num += 1
        
        
        x0 = x1
        x1 = x_aux
            

    return x1, iter_num

# Datele de intrare conform sectiunii "Rezolvare"
    
EPS = 10**(-10)
a = [-3, -1, 2]
b = [-1.5, 0.8, 3]

for i in range(len(a)):
    assert fun(a[i]) * fun(b[i]) < 0, 'Capetele intervalelor sunt gresite'


for i in range(0, len(a)):
    
    print('Metoda secantei')
    
    x0 = a[i]
    x1 = b[i]
    x_num, iter_num = metoda_secantei(fun, a[i], b[i], x0, x1, EPS)
    
    print(x_num)
    
    print('Ecuatia: x ^ 3 - x ^ 2 - 4 * x  + 3')
    print('Intervalul: [{:.2f},{:.2f}]'.format(a[i], b[i]))
    
    print('Solutia numerica: x_num = {:.5f}'.format(x_num))
    print('Numarul de iteratii: iter_num = {}'.format(iter_num))
    
    print('-' * 50)
    
    plt.scatter(x_num, fun(x_num), s=100, c='black', marker='o')  



x = np.linspace(-3, 3 , 1000)
y = fun(x)
plt.plot(x, y, linestyle='-', linewidth=3)

plt.legend(['f(x)']) 
plt.xlabel('x')
plt.title('Metoda Secantei')
plt.grid(b = True)


plt.savefig('metoda_secantei.png')
plt.show()

