'''
 Tema 3 - Calcul Numeric
 Badea Adrian Catalin, grupa 334, anul III, Informatica
 Facultatea de Matematica si Informatica


 Python 3.7.6
 Numpy 1.18.1
 Matplotlib 3.1.3

'''

'''
Exercitiul 1.
    
    Sa se verifice daca functia admite un punct de minim unic.
    
    In caz afirmativ sa se determine folosind :
        
        a. Metoda pasului descendent
        b. Metoda gradientilor conjugati
    
    f(x, y) = 2.0 * x ^ 2 - 4.0 * x * y - 9 * x + 2.5 * y ^ 2 - 6 * y

Rezolvare.

    Derivate partiale:
    
    f(x, y)'x = 4.0 * x - 4.0 * y - 9
    f(x, y)'y = 5.0 * y - 4.0 * x - 6
    
    Matricea A :
    a[1, 1] - a[1, 2]
    a[1, 2] - a[2, 2]
        
        
    f(x, y) = 1/2 * a[1,1] * x ^ 2 + 
              1/2 * a[2,2] * y ^ 2 + 
              a[1, 2] * x * y + 
              b[1] * x - b[2] * y
    
    f(x, y) = 2.0 * x ^ 2
              - 4.0 * x * y 
              - 9 * x
              + 2.5 * y ^ 2 
              - 6 * y


    1/2 * a[1, 1] = 2.0 => a[1, 1] = 4
    1/2 * a[2, 2] = 2.5 => a[2, 2] = 5
    
    a[1, 2] = - 4.0 
    b[1] = -9
    b[2] = -6
    
    B = [-9, -6]
    
    A = [[4, -4],
        [-4, 5]]
    
    det(A, 1) = 4
    det(A, 2) = 4 * 5 - (-4) * (-4) = 20 - 16 = 4

    A este simetrica si pozitiv definita, deci admite punct de minim local si este unic.


'''

import numpy as np

from matplotlib import pyplot as plt

ε = 10**(-10)



def metodaPasuluiDescendent(n, ε, a, b):
    
    
    x_num = np.zeros(n)
    r_k = b - np.matmul(a, x_num)

    k = 0
    
    while np.linalg.norm(r_k, ord = 2) > ε:
        
        # pasul algoritmului
        k += 1
        
        # r este valoarea la momentul k pe care o folosim
    
        r_k = b - np.matmul(a, x_num)
        
        α1_k = np.matmul(np.transpose(r_k), r_k) 
        α2_k = np.matmul(np.matmul(np.transpose(r_k), a), r_k)
        
        # α raportul alpha
        
        α_k = α1_k / α2_k
        
        # Se adauga la solutie directia si coeficientul acestuia in functie de ultimele valori
        
        x_num = x_num + α_k * r_k

        #print(x_num)

    return x_num
    
def metodaGradientilorConjugati(n, ε, a, b):
    
    
    x_num = np.zeros(n)
    r_k = b - np.matmul(a, x_num)
    d_k = r_k
    k = 0
    
    # Folosim metoda conform cursului
    
    while np.linalg.norm(r_k, ord=2) > ε:
        
        # k este pasul algoritmului
        k += 1
        α1_k = np.matmul(np.transpose(r_k), r_k) 
        α2_k = np.matmul(np.matmul(np.transpose(d_k), a), d_k)
        
        α_k = α1_k / α2_k
    
        x_num = x_num + α_k * d_k
        
        # α se foloseste pentru a calcula urmatoarea valoare r
        
        r_k1 = r_k - α_k * np.matmul(a, d_k)
        
        β_κ = np.matmul(np.transpose(r_k1), r_k1) / np.matmul(np.transpose(r_k), r_k)    
        
        # β se folosesten pentru a calcula cum vechea directie o influenteaza pe urmatoare
        
        d_k = r_k1 + β_κ * d_k                           
        
        r_k = r_k1
        
        # Avem nevoie sa retinem ultimele 2 valori ale lui r care determina directia d
        # Le calculam in functie de 
        
        
    return x_num

def grid_discret(a, b, x, y, size = 100):
    
    # Construieste un grid discret si evaleaza f in fiecare punct al gridului
    
    # size ->  Numar de puncte pe fiecare axa
    
    x1 = np.linspace(-x, x, size) # Axa x1
    
    x2 = np.linspace(-y, y, size) # Axa x2
    
    X1, X2 = np.meshgrid(x1, x2) # Creeaza un grid pe planul determinat de axele x1 si x2
    
    X3 = np.zeros((size, size))
    
    for i in range(size):
    
        for j in range(size):
        
            x = np.array([X1[i,j], X2[i,j]]) # x e vectorul ce contine coordonatele unui punct din gridul definit mai sus
            
            X3[i,j] = 0.5 * np.matmul(np.matmul(x, a), np.transpose(x)) - np.matmul(x, np.transpose(b)) # Evaluam functia in punctul x
            
    return X1, X2, X3

def grafic_f(a, b, x, y):
    
    # Construieste graficul functiei f
    
    # Construieste gridul asociat functiei
    (X1, X2, X3) = grid_discret(a, b, x, y)

    # Defineste o figura 3D
    plt.plot()
    ax = plt.axes(projection="3d")

    # Construieste graficul functiei f folosind gridul discret X1, X2, X3=f(X1,X2)
    ax.plot_surface(X1, X2, X3, rstride=1, cstride=1, cmap='winter', edgecolor='none')

    # Etichete pe axe
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    # Titlu
    ax.set_title('Graficul functiei f')

    # Afiseaza figura
    plt.show()

def linii_nivel(a, b, x, y, levels = 75):
    
    # Construieste liniile de nivel ale functiei f
    
    # Construieste gridul asociat functiei
    (X1, X2, X3) = grid_discret(a, b, x, y)
    
    # Ploteaza liniile de nivel ale functiei f
    plt.plot()
    # levels = numarul de linii de nivel

    plt.contour(X1, X2, X3, levels=levels) 
    # Etichete pe axe
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # Titlu
    plt.title('Liniile de nivel ale functiei f')
    
    # Afiseaza figura
    plt.show()


if __name__ == "__main__":
    
    b = np.asarray([9, 6], dtype = 'float64')
    a = np.asarray([[4., -4.], [-4, 5]], dtype = 'float64')
    sol1 = metodaPasuluiDescendent(2, ε, a, b)
    
    sol2 = metodaGradientilorConjugati(2, ε, a, b)
    
    print(sol1, sol2)
    
    linii_nivel(a, b, 50, 50)
    grafic_f(a, b, 50, 50)
    



'''

Exercitiul 2.
Sa se aproximeze functia cu polinomul Lagrange determinat folosind metoda Newton.

f(x) = 5 * sin(- x) + 2 * cos(5 * x) + 0.25 * x, x in [-π, π]

1. Sa se reprezinte grafic functia exacta, nodurile de interpolare alese si
aproximarea numerica obtinuta. Se va alege cel mai mic n pentru care ε < 10^(-5)
2. Sa se reprezinte intr-o figura noua eroarea de trunchiere.

'''

# Definim functia pentru exercitiul nostru
def fun(x):
   return 5 * np.sin(-x) + 2 * np.cos(5 * x) + 0.25 * x 

def metoda_newtonD(x, y, x_domain, nr_points):
    
    n = x.shape[0]
    
    q = np.zeros((n, n))
    
    # q este matricrea cu solutiile pe prima coloana
    
    for i in range(n):
        q[i, 0] = y[i]
    
    # Apoi se calculeaza solutiile pentru fiecare index necalculat, recurent
        
    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = (q[i, j - 1] - q[i - 1, j - 1]) / (x[i] - x[i - j])
    
    # solutia noastra de interpolare se afla in y_interp_newtonD
    
    y_interp_newtonD = np.zeros(nr_points)
    
    for i in range(nr_points):
        
        Pni = q[0, 0]
        
        # Pni este solutia la pasul i pentru p
        
        for k in range(1, n):
            p = 1
            for j in range(k):
                p = p * (x_domain[i] - x[j])
                                    
            Pni += q[k, k] * p
        
        y_interp_newtonD[i] = Pni

    return y_interp_newtonD
    

def calculeazaEroareaInterpolarii(y_interp, y_values, nr_points):
    
    # Functie care calculeaza eroarea maxima a interpolarii pentru codomeniul discretizat
    ε = 0 
    for i in range(nr_points):
        ε = max(ε, abs(y_interp[i] - y_values[i]))
    return ε 

if __name__ == "__main__":
    
    a = - np.pi
    b = np.pi
    interval = [a, b]  # [a, b]
    
    nr_points = 500

    # Discretizare domeniu (folosit pentru plotare)    
    
    x_domain = np.linspace(interval[0], interval[1], nr_points) 
    
    # Valorile functiei exacte in punctele din discretizare
    
    y_values = fun(x_domain) 
    
    # Afisare grafic figure
    
    plt.figure(0)
    plt.plot(x_domain, y_values, c='k', linewidth=2, label='Functia noastra')
    plt.xlabel('x')
    
    # Functia noastra 
    plt.ylabel('f(x)')
    plt.grid()
    
    ε = 10 ** (-5)
    N = 1
    
    N_values = []
    ε_for_N = []
    
    while True:
        
        N += 1
    
        # Discretizare interval, nodurile polinomului
        x_new = np.linspace(interval[0], interval[1], N + 1)  
        
        # Valorile functiei in nodurile date de client
        y_new = fun(x_new)  
    
        # Calculare discretizare polinom
        y_interp_newtonD = metoda_newtonD(x_new, y_new, x_domain, nr_points)
        
        ε_max = calculeazaEroareaInterpolarii(y_interp_newtonD, y_values, nr_points)
        
        N_values.append(N)
        ε_for_N.append(ε_max)
        
        print(ε_max)
        print(N)
        
        if ε_max < ε:
            break
        
        
    print(y_interp_newtonD)
    plt.scatter(x_new, y_new, marker= '*' , c='red', s=100, label='Date polinom')
    
    # Afisare grafic aprixomare
    plt.plot(x_domain, y_interp_newtonD, c='b', linewidth=1, linestyle='-.', label='Metoda Newton')
    
    plt.title('Interpolare Lagrange folosind metoda Newton cu diferente divizate, N={}'.format(N))
    plt.show()
    
    # Afisarea erorii de trunchiere
    plt.title('Eroarea de trunchiere pentru metoda Lagrange')
    plt.plot(N_values, ε_for_N, c='b', linewidth=1, linestyle='-.', label='Eroarea de trunchiere pentru metoda Lagrange')
    plt.scatter(N_values, ε_for_N, marker= '*' , c='red', s=100, label='ε values')
    
    plt.xlabel('N')
    plt.ylabel('Eroare')
    plt.grid()
    
    plt.show()
    
    
'''

Exercitiul 3.
Sa se aproximeze functia folosind interpolarea cu functii spline cubice.


f(x) = 7 * sin(- 2 * x) - 4 * cos(5 * x) - 2.91 * x, x in [-π, π]

a. Sa se reprezinte grafic functia, nodurile de interpolare alese, si aproximarea numerica a acestora.
Numarul de subintervale in care se va imparti domeniul N, se va alege cel mai mic
posibil, astfel incat eroarea maxima de trunchiere sa fie ε < 10^(-5)
b. Sa se reprezinte grafic eroarea de trunchiere


f'(x) = 7 * (-2) * cos (-2 * x) + 4 * 5 * sin(5 * x) - 2.91
f'(x) = -14 * cos(-2 * x) + 20 * sin(5 * x) - 2.91


'''

# Functia noastra in x
def fun_spline(x):
    return 7 * np.sin(- 2 * x) - 4 * np.cos(5 * x) - 2.91 * x

# Derivata in x
    
def fun_spline_der(x):
    return (-14) * np.cos(-2 * x) + 20 * np.sin(5 * x) - 2.91
 
# Algoritmul Thomas pentru matricea X
    
def algoritmulThomas(v, n, d):
    
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    
    # Construim vectorii a, b, c
    # https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9#file-tdmasolver-py
    
    # a este vectorul de sub diagonala
    for i in range(1, n):
        a[i] = v[i, i - 1]
    a[0] = 0
    
    # b este vectorul de pe diagonala
    for i in range(n):
        b[i] = v[i, i]
    
    # c este vectorul de deasupra diagonalei
        
    for i in range(n - 1):
        c[i] = v[i, i + 1]
    
    c[n - 1] = 0
    
    for k in range(1, n):

        m = a[k] / b[k - 1]
        
        b[k] -= m * c[k - 1]
        
        d[k] -= m * d[k - 1]
    
    p = np.zeros(n)
    
    p[n - 1] = d[n - 1] / b[n - 1]
    
    for k in range(n - 2, -1, -1):
        p[k] = (d[k] - c[k] * p[k + 1]) / b[k]
        
    return p

def splineCubic(X, Y, fda, fdb, x_domain):
    
    n = X.shape[0]
    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)
    h = np.zeros(n - 1)
    
    # h este vectorul de diferente 
    # a este initial copie a vectorului de solutii
    
    for i in range(n - 1):
        a[i] = Y[i]
        h[i] = X[i + 1] - X[i]
        
    q = np.zeros((n, n))
    
    q[0, 0] = 1
    
    q[n - 1, n - 1] = 1
    
    for i in range(1, n - 1):
        q[i, i - 1 : i + 2] = [1, 4, 1]
        
    step = (X[n - 1] - X[0]) / (n - 1)
    
    bb = np.zeros((n, 1))
    
    bb[0] = fda
    
    bb[n - 1] = fdb
    
    for i in range(1, n - 1):
        bb[i] = (3 / step) * (Y[i + 1] - Y[i - 1])

    # b sunt solutiile matricei tridiagonale
    b = algoritmulThomas(q, n, bb)

    for i in range(n - 1):
        
        # Calculam vectorii c si d pentru algoritmul urmator
        
        x1 = (3 / h[i] ** 2) * (Y[i + 1] - Y[i])
        y1 = (b[i + 1] + 2 * b[i]) / h[i]
        
        x2 = (-2 / h[i] ** 3) * (Y[i + 1] - Y[i])
        y2 =  (1 / h[i] ** 2) * (b[i + 1] + b[i])
        
        c[i] = x1 - y1
        d[i] = x2 + y2
        
    
    s = np.zeros(len(x_domain))
    
    for i in range(len(x_domain) - 1):
        
        for j in range(n - 1):
          
            if x_domain[i] >= X[j] and x_domain[i] < X[j + 1]:
            
                # Adunam la vectorul a diferentele ridicate la puterea corespunzatoare
                
                s[i] = a[j]
                s[i] += b[j] * (x_domain[i] - X[j])
                s[i] += c[j] * (x_domain[i] - X[j]) ** 2
                s[i] += d[j] * (x_domain[i] - X[j]) ** 3
                
                break  
            
    # Aplicam pasul anterior pentru fiecare solutie, mai putin pentru ultima
    s[len(x_domain) - 1] = Y[-1]
    
    return s
    

if __name__ == "__main__":

    a = -np.pi
    b = np.pi
    importinterval = [a, b] 
    
    # Daca folosim putine de discretizare o sa gasim o solutie mai slaba
    
    nr_points = 2000    
        
    ε = 10 ** (-5)
    N = 1

    # Discretizare domeniu (folosit pentru plotare)    
    x_domain = np.linspace(interval[0], interval[1], nr_points) 
    # Valorile functiei exacte in punctele din discretizare
    y_values = fun_spline(x_domain) 
    # Afisare grafic figure
    plt.figure(0)
    
    plt.plot(x_domain, y_values, c='k', linewidth=2, label='Functia noastra')
    
    plt.xlabel('x')
    # Functia noastra 
    plt.ylabel('f(x)')
    
    plt.grid()
    
    N_values = []
    ε_for_N = []
    
    while True:
        
        N += 1
    
        # Discretizare interval, nodurile polinomului
        x_new = np.linspace(interval[0], interval[1], N + 1)  
        
        # Valorile functiei in nodurile date de client
        y_new = fun_spline(x_new)  
    
        # Calculare discretizare polinom
        y_spline = splineCubic(x_new, y_new, fun_spline_der(a), fun_spline_der(b), x_domain)
        
        ε_max = calculeazaEroareaInterpolarii(y_spline, y_values, nr_points)
        
        # Tinem valorile pentru 
        N_values.append(N)
        ε_for_N.append(ε_max)
        
        print(ε_max)
        print(N)
        
        if ε_max < ε:
            break
        
        
    print(y_spline)
    plt.scatter(x_new, y_new, marker= '*' , c='red', s=100, label='Date polinom')
    
    # Afisare grafic aprixomare
    plt.plot(x_domain, y_spline, c='b', linewidth=1, linestyle='-.', label='Spline cubic')
    
    plt.title('Spline cubic, N={}'.format(N))
    plt.show()
    
    # Afisarea erorii de trunchiere
    plt.title('Eroarea de trunchiere pentru Spline cubic')
    
    plt.plot(N_values, ε_for_N, c='b', linewidth=1, linestyle='-.', label='Eroarea de trunchiere pentru metoda Lagrange')
    
    plt.scatter(N_values, ε_for_N, marker= '*' , c='red', s=100, label='ε values')
    
    plt.xlabel('N')
    plt.ylabel('Eroare')
    
    plt.grid()
    plt.show()

    
    