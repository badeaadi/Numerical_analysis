'''
 Tema 2 - Calcul Numeric
 Badea Adrian Catalin, grupa 334, anul III, Informatica
 Facultatea de Matematica si Informatica

'''

import numpy as np

'''
Exercitiul 1. Sa se verifice daca sistemul dat admite soluttie unica 
si in caz afirmativ sa se determine solutia folosind 
Metoda Gauss cu pivotare totala.

Sa se verifice daca sistemul admite solutie unica si in caz afirmativ 
sa se determine solutia folosind metoda Gauss cu pivotare totala.


Rezolvare.

Vom folosi metoda prezentata in curs.
Initial vom verifica ca matricea este patratica si ca vectorul b se potriveste in lungime.

La fiecare pas (1 .. n - 1) al algoritmului, alegem ca pivot elementul curent cu valoarea
absoluta cea mai mare dintr-o anumita submatrice

'''


EPS = 0.000001

def gaussPivotareTotala(a, b):
    
    """ (Optional) Verifica daca vatricea este patratica + compatibilitatea cu vectorul b. """
    
    assert np.asarray(a).shape[0] == np.asarray(a).shape[1], 'matricea introdusa nu este patratica!'
    assert np.asarray(a).shape[0] == np.asarray(b).shape[0], 'matricea introdusa si vectorul b nu se potrivesc!'
      
    v = a
    
    # α este lungimea matricei initiale
    α = len(v)  
    x = np.zeros(len(a))
    
    for i in range(0, α):
       v[i].append(b[i]) 
       
    # Lungimea matricei extinse
    β = len(v[i])  
    
    # Vectorul de pozitie al solutiilor, ne va ajuta in reconstructia solutiilor
    pos = []
    for i in range(0, α):
        pos.append(i)
    
    for k in range(α):   
    
       
        # Gasim elementul (pivotul), elementul maxim in valoare absoluta din submatricea urmatoare
        # Acesta are indicii psol, msol la final
        
        maxim = -1
        for p in range(k, α):
            for m in range(k, α):
                if abs(v[p][m]) > maxim:
                   
                   maxim = abs(v[p][m])
                   psol = p;
                   msol = m
      
        assert maxim > EPS  , 'Sistem incompatibil sau compatibil nedeterminat'
        
        
        # Daca p-ul gasit este diferit de k, schimbam liniile
        if psol != k:
           v[psol], v[k] = v[k], v[psol]
      
      
        # Daca m-ul gasit este diferit de k, schimba coloanele
        
        if msol != k :
           
           pos[msol], pos[k] = pos[k], pos[msol]
           for i in range(0, α):
               v[i][msol], v[i][k] = v[i][k], v[i][msol]
            
        for l in range(k + 1, α):
            rap = a[l][k] / a[k][k]
            
            # Scadem din linia l linia k inmultita cu raportul
            for i in range(0, β):
                v[l][i] -= rap * v[k][i]
         

    assert v[α - 1][α] != 0, 'Sistem compatibil sau compatibil nedeterminat'
    
    
    # Reconstruim solutia folosind metoda substitutiei ascendente
    for i in range(α - 1, -1, -1):
        for j in range(β - 2, β - (α - i + 1), -1) :
            v[i][β - 1] -= v[i][j] * x[j]
        
        sol = v[i][β - 1] / v[i][β - (α - i + 1)]
        
        x[i] = sol
    
    # Permutam solutia in ordinea care trebuie (au fost permutate coloane)
    
    y = x
    
    for i in range(0, α):
        for j in range(0, α):
            if (pos[i] < pos[j]):
                pos[i], pos[j] = pos[j], pos[i]
                y[i], y[j] = y[j], y[i]
        
    return y


if __name__ == "__main__":
    
    a = [[0, -1, -9, -6],
         [5, 1, 8, 3],
         [2, 3, 6, -8],
         [3, 8, -9, 7]] 
    
    b = [-85, 77, 0, 38]
    
    # Apelam functia cu parametri corespunzatori
    
    print(gaussPivotareTotala(a, b))
    

'''

Exercitiul 2. Verificati daca matricea B este inversabila si in caz afirmativ aplicati 
Metoda Gauss pentru determinarea inversei.


Rezolvare. Pentru verificarea invserabilitatii, putem folosi functia determinantOfMatrix care calculeaza recursiv, naiv,
dupa prima linie, determinantul matricei. 
Altfel, am putea folosi functia de calcul al determinantului din numpy


Daca determinantul este diferit de 0, atunci cautam inversa 
folosing metoda Gauss Jordan cu esalonarea matricei.
    
'''  



# Functia care calculeaza submatricea fara liniile p, q si o pune in temp

def getSubMatrix(mat, temp, p, q, n): 
    
    i = 0
    j = 0   
    for row in range(n):  
        for col in range(n): 
             
            if (row != p and col != q) : 
                  
                temp[i][j] = mat[row][col] 
                j += 1
                if (j == n - 1): 
                    j = 0
                    i += 1
  
# Functia care calculeaza determinantul matricei recursiv
def determinantOfMatrix(mat, n): 
    det = 0 
  
    if (n == 1): 
        return mat[0][0] 
          
    # Matricea temporara
    temp = [[0 for x in range(n)]  
               for y in range(n)]  
  
    # Semn initial - plus (pentru primul determinant din calcul)
    sign = 1
  
    for f in range(n): 
          
        getSubMatrix(mat, temp, 0, f, n)
        
        # Adunam la determinantul curent determinantele submatricelor factor (dupa prima linie)
        
        det += (sign * mat[0][f] *
              determinantOfMatrix(temp, n - 1)) 
  
        sign = -sign 
        
        # Dupa fiecare adunare schimbam semnul
        
    return det

# Functie care returneaza daca matricea este inversabila daca determinantul este diferit de 0
def isInvertible(mat, n): 
    
    if (abs(determinantOfMatrix(mat, n)) < EPS): 
        return False
    else: 
        return True


def inverseGaussJordan(mat, n):
    
    a = np.zeros((n,2*n), np.float32)

    # Augmentare gauss-jordan 
    for i in range(n):
        for j in range(n):
            a[i][j] = mat[i][j]
    
    for i in range(n):
        a[i][i + n] = 1
    
    # Aplicam eliminarea Gauss Jordan
    
    for i in range(n):
        
        # Daca a[i][i] este diferit de 0, il folosim ca pivot
        # Altfel, alegem ca pivot un element de pe coloana i si interschibam cu linia i
    
        if abs(a[i][i]) < EPS:
            
            for j in range(i + 1, n):
                if abs(a[j][i]) > EPS:
                    break
            
            # Interschimbam liniile i si j 
            a[[i, j]] = a[[j, i]]
            
        # Scadem linia i din fiecare celelalte.
        for j in range(n):
            if i != j:
      
                ratio = a[j][i] / a[i][i]
                
                for k in range(2*n):
                    
                    a[j][k] = a[j][k] - ratio * a[i][k]
        
    for i in range(n):
        
        divisor = a[i][i]
        
        for j in range(2*n):
            
            a[i][j] = a[i][j] / divisor
        
    # Reconstruim solutia in b
    
    b = np.zeros((n, n), np.float32)
    
    for i in range(n):
        for j in range(n):
            b[i][j] = a[i][j + n]
    
    return b


if __name__ == "__main__":
    
    a = [[0, 3, 1, -1],
         [3, 8, -10, 5],
         [-5, -8, 0, 0],
         [4, 5, -4, 7]] 
    
    if isInvertible(a, 4):
        print(inverseGaussJordan(a, 4))
    else:
        print ("Matricea nu este inversabila")

        
''' 
Exercitiul 3.
Sa se verifice daca sistemul admite solutie unica si in caz afirmativ 
sa se determine solutia folosind factorizarea LU cu pivotare partiala.

Rezolvare. Vom transforma matricea initiala intr-o pereche de doua matrice, L si U
pentru care L este inferior triunghiulara U este superior triunghiulara, si L * U = matricea initiala
Descompunerea lor este unica.

Apoi, A * X = B echivalent cu L * U * X = B
Notam U * X cu Y, de aici rezulta ca L * Y = B,

determinam Y folosind substitutie ascendenta
revenim la notatie si determinam X folosind substitutie descendenta

'''


def LUFactor(a):
    
    n = a.shape[0]
    
    # l este matricea identitate
    
    l = np.zeros((n, n), np.float64)
    for i in range(n):
        l[i][i] = 1
        
    w = np.arange(0,n)
    
    for k in range(0, n - 1):

        p = np.argmax(abs(a[k:n, k])) + k
        
        # Indicele p este linia argumentului maxim
        
        assert abs(a[p][k]) > EPS, 'Nu admite factorizare LU'
        
        # Daca p este diferit de k, le interschimbam in matrice
            
        if p != k:
            a[[k,p], :] = a[[p,k], :]
            w[p], w[k] = w[k], w[p]
       
            if k > 0 :
                # Interschimbam subliniile situate sub diagonala principala intre ele
                
                for i in range(0, k):
                    l[k][i], l[p][i] = l[p][i], l[k][i]

        
        for i in range(k + 1, n):          
            
            l[i,k] = a[i,k] / a[k,k]      
            
            for j in range(0,n):      
            
                a[i,j] -= l[i,k]* a[k,j] 

    return (l, a, w)


# Metoda substitutiei ascendente

def sub_asc(a, b):
    
    n = a.shape[0]
    
    # Initializam tabloul solutiilor
    x_num = np.zeros(n)
    
    # Prima solutie
    
    x_num[0] = b[0] / a[0][0]
    
    # Reconstruim pe rand solutiile de jos in sus
    
    for i in range(1, n):
        
        # Gasim suma pe linie pentru toate coloanele diferite de cea pe care urmeaza sa o calculam
        for j in range(i) :
            x_num[i] -= a[i][j] * x_num[j]
        
        x_num[i] = (b[i] + x_num[i]) / a[i][i]
    
    return x_num
    

def sub_desc(a, b):
    
    
    n = a.shape[0]
    
    # Initializam tabloul solutiilor
    x_num = np.zeros(n)
    
    # Prima solutie (in partea de jos)
    
    x_num[n - 1] = b[n - 1] / a[n - 1][n - 1]
    
    # Reconstruim pe rand solutiile de sus in jos
    
    for i in range(n - 2, -1, -1):
        
        
        # Gasim suma pe linie pentru toate coloanele diferite de cea pe care urmeaza sa o calculam
        for j in range(i + 1, n) :
            x_num[i] -= a[i][j] * x_num[j]
        
        x_num[i] = (b[i] + x_num[i]) / a[i][i]
    
    return x_num




if __name__ == "__main__":
    
    a = [[0, -1, 9, 1],
         [8, 0, 5, 7],
         [-2, 9, 5, 0],
         [1, 0, 2, -3]] 
    
    b = [38, 71, 43, -5]      
    b = np.array(b, np.float64)

    # l, u sunt matricele mentionate in rezolvare, pentru care l * u = a
    
    l, u, w = LUFactor(np.asarray(a, np.float64))
    
    bp = np.zeros(len(b))
    for i in range(len(b)):
        bp[i] = b[w[i]]
    
    
    y = sub_asc(l, bp)
    print(sub_desc(u, y))

'''
Exercitiul 4.
Sa se verifice daca matricea C admite facrotizare Choleskuy si in caz afirmativ
Sa se determine aceasta factorizare.

Rezolvare.
Daca matricea este pozitiv definita si este simetrica, 
aceasta admite factorizare Cholesky

Pentru o matrice A, cautam o matrice L inferior triunghiulara, 
astfel incat L * L(transpus) = A
'''        
        


def Cholesky(a):
    
     α = a[0][0]
     
     # Verificam ca A este pozitiv definita
     
     assert α  > 0, 'A nu este pozitiv definit'
     
     λ = a.shape[0]
     
     
     # Verificam proprietatea de simetricitate a elementelor
     
     symmetric = True
     for i in range(λ):
         for j in range(λ):
             if (a[i][j] != a[j][i]) :
                 symmetric = False
                 
     assert symmetric == True, 'Matricea nu este simetrica'
     
     l = np.zeros((λ, λ), np.float64)
     
        
     l[0][0] = np.sqrt(α)
     
     for i in range(1, λ):
         l[i][0] = a[i][0] / l[0][0] 

     for k in range(1, λ):
         
        # Calculam suma patratelor in Σ
         
        Σ = 0
        for j in range(0, k):
            Σ += l[k][j] * l[k][j]
        
        α = a[k][k] - Σ
        
        # Verificam ca la fiecare moment α este mereu pozitiv calculat
        
        assert α  > 0, 'A nu este pozitiv definit'
        
        
        # Elementul de pe diagonala principala este mereu dat de α corespunzator.
        
        l[k][k] = np.sqrt(α)
        
        
        for i in range(k + 1, λ):
            
            Σ = 0
            for j in range(0, k):
                
               Σ += l[i][j] * l[k][j]
               
            l[i][k] = (1.0 / l[k][k]) * (a[i][k] - Σ)

     return l
         

if __name__ == "__main__":
    
    a = [[49, -28, 35, -28],
         [-28, 65, -6, 2],
         [35, -6, 45, 8],
         [-28, 2, 8, 165]] 

    cholesky = Cholesky(np.array(a, np.float64))
    
    # Pentru verificarea rezultatelor (ar trebui sa fie exact aceeasi matricea ca cea initiala)
    print(np.matmul(cholesky, np.transpose(cholesky)))


    
        
        
        
    
        
        
        