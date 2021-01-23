
'''
 Tema 4 - Calcul Numeric
 Badea Adrian Catalin, grupa 334, anul III, Informatica
 Facultatea de Matematica si Informatica


 Python 3.7.6
 Numpy 1.18.1
 Matplotlib 3.1.3

'''



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
Ex. 1 (4.5 puncte)

Sa se aproximeze a doua derivata a functiei folosind metoda diferentelor finite.

1.  Sa se reprezinte grafic derivata a doua exacta a functiei (1) ¸si aproximarea numerica obtinuta. 
Numarul de puncte al discretizarii intervalului, N, se va alege cel mai mic posibil,
astfel incat eroarea maxima de trunchiere sa satisfaca relatia e ≤ 1e − 5.

2. Sa se reprezinte intr-o figura noua eroarea de trunchiere.

f(x) = cos(0.7 * x), x in [-π/2, π]

f'(x) = -(0.7) * sin(0.7 * x)

f''(x) = (-0.7) * 0.7 * cos(0.7 * x) = -0.49 * cos(0.7 * x)

'''

# Functia noastra 

def fun(x):
    return np.cos(0.7 * x)

# Derivata functiei, simbolic

def fun_first_derivative(x):
    return -0.7 * np.sin(0.7 * x) 

# A doua derivata a functiei, simbolic
    
def fun_second_derivative(x):
    
    return -0.49 * np.cos(0.7 * x)


def second_derivative_numeric_derivation(x):
    
    h = x[1] - x[0]
    
    # Imaginea functiei dupa  liniarizare
    
    y_img = fun(x)
    
    # Imaginea dupa derivarea numerica cu diferente centrale
    
    y_img_computed = np.zeros(len(y_img) - 2)
   
    # Aplicam formula diferentelor centrale
    
    for i in range(1, len(x) - 1):
        y_sum = (y_img[i + 1] -  2  *  y_img[i]  +  y_img[i - 1])
        
        y_img_computed[i - 1] =  y_sum / ((x[i + 1] - x[i]) ** 2)

    return y_img_computed

    

    



if __name__ == "__main__":
    
    # Eroarea maxima de trunchiere

    ε = 1e-5
    
    # Capetele intervalului functiei
    
    a = -np.pi/2 
    b =  np.pi
    
    # Presupunem N = numarul de puncte ale liniarizarii, cat mai mic
    
    N = 19
    ε_now = 1
    
    Ns = []
    ε_list = []
    
    while (ε_now > ε) :    
        
        N += 1
        # Liniarizam spatiul pentru N puncte 
        x_lin = np.linspace(a, b, N)
        
        # Extragem aproximarea pentru numarul de puncte curent
        
        y_img_computed = second_derivative_numeric_derivation(x_lin)
       
        # Extragem imaginea celei de a doua derivate
        
        y_img = fun_second_derivative(x_lin)
        
        ε_now = 0
        
        for i in range(1, len(y_img) - 1):
            ε_now = max(ε_now, abs(y_img[i] - y_img_computed[i - 1]))    
        
        
        print('Pentru N : {}, valoarea erorii este : {}'.format(N, ε_now))
        
        Ns.append(N)
        ε_list.append(ε_now)
        
    
    print("Eroare este mai mica decat {}".format(ε))
    
    mpl.style.use('seaborn')
    x_lin = np.linspace(a, b, N)
    y_simbolic = fun_second_derivative(x_lin[1:-1])
    y_numeric = second_derivative_numeric_derivation(x_lin)
    
    plt.suptitle('Aproximarea derivatei a II-a')
    plt.plot(x_lin[1:-1], y_simbolic, '--', label='Calcul simbolic')
    plt.plot(x_lin[1:-1], y_numeric, '--',label='Calcul numeric')
    plt.legend()
    plt.savefig("Aproximariea derivatei secundare.png", dpi = 300)
    plt.show()
            
    
    plt.suptitle("Eroare de trunchiere pentru aproximare")
    plt.plot(Ns, ε_list, 'o', label = 'Eroare in functie de N')
    plt.legend()
    plt.savefig("Eroare de trunchiere pentru aproximarea derivatei a doua.png")
        
        

'''

2. Creati functia integrare care calculeaza valoarea aproximata a functiei 
    I(f) = Integrala de la a la b din f(x) dx conform formulelor de cuadratura
    sumata ale dreptunghiului, trapezului si Simpson, si are ca date de intrenare:
        
        i. functia de integrare
        ii. vectorul x, o divizune a intervalului [a, b]
        iii. sirul de caractere metoda din {'dreptunghi', 'trapez', 'Simpson'}
        
        
    f(x) = 1/ (σ * sqrt((2 * π)))  * e ^ (- x ^ 2 / 2 * σ ^ 2)  
    
    Vom defini o subdiviziune echidistanta si vom aplica metodele
    de cuadratura sumata pentru definirea integralei din punct de vedere numeric.
    
    
    
    
    Bonus.
    
    Integrare simbolica.
    
    σ = 1.0 
    
    I(f(x)) = 1 / (σ * sqrt((2 * π))  * I ( e ^ ((- x ^ 2 / 2 * σ ^ 2)))
            
    1 / (σ * sqrt((2 * π)) = constanta 
    
    I(e ^ ((- x ^ 2) / 2 * σ ^ 2))) = integrala gaussiana
    
    care, daca este nedefinita, nu are solutie
    
    dar pentru varianta definita, 
    
    solutia (de la -inf la inf) este : 
    
            sqrt(π / a), unde a = (1 / (2 * σ ^ 2))
            
    Deci, I = 1 / (σ * sqrt(2 * π)) *  sqrt(π / (1 / (2 * σ ^ 2)))
            = 1 / (σ * sqrt(2 * π)) *  sqrt(π * 2 * σ ^ 2)
            = 1 / (σ * sqrt(2 * π)) * (σ * sqrt(2 * π)) = 1
            
'''

# Functia din integrala

def function(x, σ):
    return 1/(σ * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / (2 * (σ ** 2)))


def numeric_integration(fun, x_lin, method, σ) :
    
    # Imaginea functiei pentru σ dat
    
    y_lin = function(x_lin, σ)
    
    # h este marimea unei subdiviziuni
    h = x_lin[1] - x_lin[0]
    
    # Formula de cuadratura sumata pentru metoda dreptunghiului
    
    # Domeniul functiei este divizat echidistant
    
    # Aplicam pe toata subdiviziunea
    
    
    # Metoda dreptunghiului
    
    if method == 'dreptunghi':
        sol = 2 * h * np.sum(y_lin[::2])
        
    
    # Metoda trapezului
            
    elif method == 'trapez':
        sol = h/2 * (y_lin[0] + 2 * np.sum(y_lin[1:-2]) + y_lin[-1])
        
    # Metoda simpson
        
    elif method == 'simpson':
        sol = h/3 * (y_lin[0] + 4 * np.sum(y_lin[1:-1:2]) + 2 * np.sum(y_lin[2:-1:2]) + y_lin[-1])

    return sol
    

if __name__ == "__main__":
    
    # Constantele pentru problema noastra
    
    a = -10
    b = 10
    σ = 1.0
    
    N_min = 5
    N_max = 30
    
    # Solutiile pentru integrarea numerica pentru fiecare N, pentru fiecare metoda
    sol_dreptunghi = []
    sol_trapez = []
    sol_simpson = []
    
    # Solutiile pentru N 
    
    Ns = []
    
    for N in range(N_min, N_max):    
        
        x_lin = np.linspace(a, b, N)
        
        y_dreptunghi = numeric_integration(function, x_lin, 'dreptunghi', σ)
        y_trapez = numeric_integration(function, x_lin, 'trapez', σ)
        y_simpson =  numeric_integration(function, x_lin, 'simpson', σ)
        
        
        Ns.append(N)
        sol_dreptunghi.append(y_dreptunghi)
        sol_trapez.append(y_trapez)
        sol_simpson.append(y_simpson)
        
        
        
    mpl.style.use('seaborn')
    plt.figure(dpi=300)
    
    plt.suptitle('Aproximarea valorii integralei folosind diferite formule de cuadratură sumata')
    
    plt.plot(Ns, sol_dreptunghi, label='Dreptunghi')
    plt.plot(Ns, sol_trapez, label='Trapez')
    plt.plot(Ns, sol_simpson, label='Simpson')
    
    plt.savefig("Aproximariea valorii integralei numerice.png", dpi = 300)
    
    plt.hlines([1], xmin=N_min, xmax=N_max - 1, linestyle='-', color='red')
    plt.legend()
    plt.show()
    
    
    
    
    