"""
    This file contains the code constructions.

    The code constructions are the following:
        1.Stabilizer codes constructed by the canonical construction.
        2.CSS code constructed by the QC-LDPC-HGP construction.
        3.CSS code constructed by the Bivariate-bycicle codes.

"""


import numpy as np
from bposd.css import css_code
class StabilizerCode():
    def __init__(self,h) -> None:
        self.h = h
        if not self.check_validity():
            raise ValueError('This stabilizer code is not valid.')
        self.n = self.h.shape[1]//2

    def check_validity(self):
        if not isinstance(self.h, np.ndarray):
            return False
        if self.h.shape[1]%2 != 0:
            return False
        n = self.h.shape[1]//2
        omega = np.block([
            [np.zeros((n,n)),np.eye(n)],
            [np.eye(n),np.zeros((n,n))]    
        ])
        if not ((self.h @ omega @ self.h.T) %2 == 0).all():
            return False

        return True
    
class CSSCode(StabilizerCode):
    def __init__(self,hx=None,hz=None) -> None:
        self.hx = np.array(hx)
        self.hz = np.array(hz)
        self.h = np.block([
            [hz, np.zeros(hz.shape)],
            [np.zeros(hx.shape), hx]
        ])
        if not self.check_validity():
            raise ValueError('This CSS code is not valid.')
        qcode = css_code(self.hx, self.hz)
        self.lx = np.array(qcode.lx.toarray())
        self.lz = np.array(qcode.lz.toarray()) # logical operators
        self.k = qcode.K
        self.n = qcode.N
        
        
    
    def check_validity(self):
        if not isinstance(self.hx, np.ndarray)or not isinstance(self.hz, np.ndarray):
            print('not np.array')
            return False
        if self.hx.shape[1]!=self.hz.shape[1]:
            print('n is not equal')
            return False
        if (((self.hz @ self.hx.T)%2) != 0).any():
            print('some x and z stabilizers do not commute')
            return False
        return True



class CodeConstructor():
    """
        This is the class for code construction.
        now we support the following methods:
            1. Canonical construction.(for Evolutionary algorithm, general stabilizer codes)
            2. QC-LDPC-HGP construction.
            3. Bivariate-bycicle construction.
            4. Rotated Surface code construction
            5. PG-LDPC-HGP construction
    """
    def __init__(self,method='qc-ldpc-hgp',para_dict=None) -> None:
        """
            method(str): The method of the code construction, 'canonical', 'qc-ldpc-hgp', 'bivariate-bycicle','rotated-surface','qc-gb'.

            
        """
        self.method = method
        self.para_dict = para_dict
        self.n = 0
        self.nx = 0
        self.nz = 0
        self.k = None
        
        if not self.check_parameters_validity():
            raise ValueError('The parameters are not valid.')
        
    def check_parameters_validity(self):
        if self.method == 'canonical' or self.method == 'canonical_css':
            return self.check_canonical_parameters_validity()
        elif self.method == 'qc-ldpc-hgp' or self.method == 'pg-ldpc-hgp':
            return self.check_qc_ldpc_hgp_parameters_validity()
        # elif self.method == 'bivariate-bycicle':
        #     return self.check_bivariate_bycicle_parameters_validity()  
        elif self.method == 'rotated-surface':
            return True
        elif self.method in ['qc-gb','bivariate-bycicle','bb','symmetric-qc-gb']:
            if self.method in ['bb','bivariate-bycicle']:
                try:
                    self.n = self.para_dict['l'] * self.para_dict['g']
                    self.nx = self.n
                    self.nz = self.n
                except:
                    raise ValueError('failing setting n,nx,nz')
            if 'g' in self.para_dict.keys():
                if type(self.para_dict['g']) == type(int(1)):
                    return True
            elif 'ga' in self.para_dict.keys() and 'gb' in self.para_dict.keys():
                if type(self.para_dict['ga']) == type(int(1)) and type(self.para_dict['gb']) == type(int(1)):
                    return True
            return False
        else:
            raise ValueError('The method is not supported.')
    def construct(self,parameters):
        if self.method == 'canonical':
            return self.canonical_construction(parameters)
        elif self.method == 'canonical_css':
            return self.canonical_construction(parameters,CSS=True)
        elif self.method == 'qc-ldpc-hgp':
            return self.qc_ldpc_hgp_construction(parameters)
        elif self.method == 'pg-ldpc-hgp':
            return self.pg_ldpc_hgp_construction(parameters)
        # elif self.method == 'bivariate-bycicle':
        #     return self.bivariate_bycicle_construction(parameters)
        elif self.method == 'rotated-surface':
            return self.rotated_surface_construction(parameters)
        elif self.method == 'qc-gb':
            return self.quasi_cyclic_generalized_bicycle_code(parameters)
        elif self.method == 'bivariate-bycicle' or self.method == 'bb':
            return self.arbitrary_bivariate_bicycle_code(parameters)
        elif self.method == 'symmetric-qc-gb':
            return self.symmetric_quasi_cyclic_generalized_bicycle_code(parameters)
        else:
            raise ValueError('The method is not supported.')

    def check_canonical_parameters_validity(self):
        """
            For each element in canonical construction requires the following parameters:
                1. n(int): The number of qubits.
                2. k(int): The number of logical qubits.
                3. r(int): The number of stabilizer generators at least have one x-operator.
                and s+r = n-k
        """
        for index in self.para_dict.keys():
            if type(self.para_dict[index])!=int or self.para_dict[index]<=0:
                return False
            if not index in {'n','k','r'}:
                return False
        if self.para_dict['n']-self.para_dict['k'] < self.para_dict['r']:
            return False
        self.n = self.para_dict['n']
        self.k = self.para_dict['k']
        self.nx = self.para_dict['r']
        self.nz = self.n-self.k-self.nx
        return True
    
    def check_qc_ldpc_hgp_parameters_validity(self):
        """
            For each element in QC-LDPC-HGP construction requires the following parameters:
                1. p(int): The number of rows of the matrix M.
                2. q(int): The number of columns of the matrix M.
                3. m(int): The size of the quasi-cyclic matrix.
                4*. p_2(int): The number of rows of the matrix M_2.  (* means optional)
                5*. q_2(int): The number of columns of the matrix M_2.
                6*. m_2(int): The size of the quasi-cyclic matrix M_2.
        """
        for index in self.para_dict.keys():
            if type(self.para_dict[index])!=int or self.para_dict[index]<=0:
                return False
        for i in  {'m','p','q'}:
            if not i in self.para_dict.keys():
                return False
        if 'p_2' in self.para_dict.keys():
            if not 'q_2' in self.para_dict.keys():
                return False
            if not 'm_2' in self.para_dict.keys():
                return False
            self.n = self.para_dict['m']*self.para_dict['m_2']*(self.para_dict['p']*self.para_dict['p_2']+self.para_dict['q']*self.para_dict['q_2'])
            self.nx = self.para_dict['m']*self.para_dict['m_2']*self.para_dict['p']*self.para_dict['q_2']
            self.nz = self.para_dict['m']*self.para_dict['m_2']*self.para_dict['p_2']*self.para_dict['q']
        else:
            self.n = self.para_dict['m']**2*(self.para_dict['p']**2 + self.para_dict['q']**2)
            self.nx = self.para_dict['m']**2*self.para_dict['p']*self.para_dict['q']
            self.nz = self.nx
        
 
        
        return True
    
    def check_bivariate_bycicle_parameters_validity(self):
        # print('check bb')
        for index in self.para_dict.keys():
            if type(self.para_dict[index])!=int or self.para_dict[index]<0:
                return False
        for i in  {'l','m'}:
            if not i in self.para_dict.keys():
                return False
        self.n = 2*self.para_dict['l'] * self.para_dict['m']
        self.nx = self.n//2
        self.nx = self.nx
        return True


    
    def rotated_surface_construction(self,p):
        """
        p*p rotated surface code
        """
        def index(row, col,reversed=False):
            if row<0 or row>=p or col<0 or col>=p:
                return -1
            if reversed:
                return row + col * p
            return row * p + col
        if p % 2 == 0:
            raise ValueError("p must be an odd number.")
        N = int(p * p)
        stabilizers_num = int((p-1)*(p-1)/2+p-1)
        Hx = np.zeros((stabilizers_num, N), dtype=int)
        Hz = np.zeros((stabilizers_num, N), dtype=int)
        

        x_stabilizers = [[] for i in range(stabilizers_num)]
        z_stabilizers = [[] for i in range(stabilizers_num)]
        for i in range(p+1):
            for j in range((p-1)//2):
                if index(i-1,j+(i+1)%2)!= -1:
                    x_stabilizers[i*(p-1)//2+j].append(index(i-1,2*j+(i+1)%2))
                if index(i-1,j+(i+1)%2+1)!= -1:
                    x_stabilizers[i*(p-1)//2+j].append(index(i-1,2*j+(i+1)%2+1))
                if index(i,j+(i+1)%2)!= -1:
                    x_stabilizers[i*(p-1)//2+j].append(index(i,2*j+(i+1)%2))
                if index(i,j+(i+1)%2+1)!= -1:
                    x_stabilizers[i*(p-1)//2+j].append(index(i,2*j+(i+1)%2+1))
        for i in range(p+1):
            for j in range((p-1)//2):
                if index(i-1,j+(i)%2,reversed=True)!= -1:
                    z_stabilizers[i*(p-1)//2+j].append(index(i-1,2*j+(i)%2,reversed=True))
                if index(i-1,j+(i)%2+1,reversed=True)!= -1:
                    z_stabilizers[i*(p-1)//2+j].append(index(i-1,2*j+(i)%2+1,reversed=True))
                if index(i,j+(i)%2,reversed=True)!= -1:
                    z_stabilizers[i*(p-1)//2+j].append(index(i,2*j+(i)%2,reversed=True))
                if index(i,j+(i)%2+1,reversed=True)!= -1:
                    z_stabilizers[i*(p-1)//2+j].append(index(i,2*j+(i)%2+1,reversed=True))

        for i in range(stabilizers_num):
            for j in x_stabilizers[i]:
                Hx[i,j] = 1
            for j in z_stabilizers[i]:
                Hz[i,j] = 1
                
        return CSSCode(Hx, Hz)
    
    def canonical_construction(self,bitstring,CSS=False) -> StabilizerCode:
        """
            Construct the stabilizer code by the canonical construction.

            Args:
                parameters(np.array): The parameters of the code.

            Returns:
                stabilizer_code(StabilizerCode): The stabilizer code.
        """
        
        n = self.n
        k = self.k
        r = self.para_dict['r']
        s = n-k-r
        
        if CSS == False:
            if len(bitstring)!=((n-r)*r+k*(n-k)+(1+r)*r//2):
                raise ValueError('The length of the bitstring is invalid.')
            for i in bitstring:
                if i!=0 and i!=1:
                    raise ValueError("The bitstring's value is invalid.")
            A = bitstring[:r*(n-r)].reshape((r,n-r))
            A_1 = A[:,:s]
            A_2 = A[:,s:]
            C = bitstring[r*(n-r):r*(n-r)+k*(n-k)].reshape((n-k,k))
            C_1 = C[:r]
            C_2 = C[r:]
            M = np.zeros((r,r))
            # Reconstruct the symmetric matrix M
            p=0
            for i in range(r):
                for j in range(i,r):
                    
                    M[i,j] = bitstring[r*(n-r)+k*(n-k)+p]
                    M[j,i] = M[i,j]
                    p += 1
            if s!= 0:
                D = (A_1.T + C_2@A_2.T)%2
                B = (C_1@A_2.T + M)%2

                H = np.block([
                    [np.eye(r), A_1, A_2, B, np.zeros((r,s)),C_1],
                    [np.zeros((s,r)), np.zeros((s,s)),np.zeros((s,k)),D,np.eye(s),C_2]
                ])
            else:
                

                B = (C_1@A_2.T + M)%2

                H = np.block([
                    np.eye(r), A_2, B,C_1
                ])

            return StabilizerCode(H)
        else:
            if len(bitstring)!=((n-r)*r+s*k):
                raise ValueError('The length of the bitstring is invalid.')
            for i in bitstring:
                if i!=0 and i!=1:
                    raise ValueError("The bitstring's value is invalid.")
            A = bitstring[:r*(n-r)].reshape((r,n-r))
            A_1 = A[:,:s]
            A_2 = A[:,s:]
            C = bitstring[r*(n-r):r*(n-r)+k*(n-k)].reshape((s,k))
            D = (A_1.T + C@A_2.T)%2
            hx = np.block([
                D, np.eye(s), C
            ])
            hz = np.block([
                np.eye(r), A_1, A_2
            ])
            return CSSCode(hx,hz)



    def qc_ldpc_hgp_construction(self,M,M_2 = None ,form = None) -> CSSCode:
        """
            Construct the CSS code by the QC-LDPC-HGP construction.

            Args:
                M(np.ndarray): The parameters of the code.M is a np.ndarray
                M_2*(np.ndarray): The parameters of the code.M_2 is a np.ndarray

            Returns:
                CSS_code(CSSCode): The CSS code.
        """
        if M_2 is None:
            H1 = self.ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
            H2 = self.ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
        else:
            H1 = self.ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
            H2 = self.ldpc_construction(self.para_dict['p_2'],self.para_dict['q_2'],self.para_dict['m_2'],M_2)
        r1,n1 = H1.shape
        r2,n2 = H2.shape
        # HX = [H1 ⊗ In2 | Ir1 ⊗ H2.T]
        # HZ = [In1 ⊗ H2 | H1.T ⊗ Ir2]
        HX_left = np.kron(H1, np.eye(n2))
        HX_right = np.kron(np.eye(r1), H2.T)
        HX = np.hstack((HX_left, HX_right))

        HZ_left = np.kron(np.eye(n1), H2)
        HZ_right = np.kron(H1.T, np.eye(r2))
        HZ = np.hstack((HZ_left, HZ_right))

        
        return CSSCode(HX,HZ)
  
    def ldpc_construction(self,p,q,m,M):
        # print(M)
        if len(M)!=p*q:
            raise ValueError('The shape of M is invalid!')
        M = M.reshape(p,q)
        H = np.zeros((p*m,q*m))

        # Define the base cyclic shift matrix S (m x m)
        S = np.zeros((m, m))
        for i in range(m - 1):
            S[i, i + 1] = 1
        S[m - 1, 0] = 1
        # Construct H using the information from M
        for i in range(p):
            for j in range(q):
                # Get the value from matrix M
                shift = M[i, j] % (m+1)
                
                # Create the shifted version of S based on the value in M
                if shift == 0:
                    H_ij = np.zeros((m,m))  # Zero matrix for shift 0
                else:
                    H_ij = np.linalg.matrix_power(S, shift)
                
                # Place H_ij in the corresponding block of H
                H[i * m: (i + 1) * m, j * m: (j + 1) * m] = H_ij
        return H

    def bivariate_bycicle_construction(self,parameters) -> CSSCode:
        '''
        para_dict={
            'm'=int,  # n = 2lm
            'l'=int,  
            parameters = [boolean,boolean,boolean,boolean,boolean,boolean,int,int,int,int,int,int]
        }
        '''
        m = self.para_dict['m']
        l = self.para_dict['l']
        Sm = np.zeros((m,m))
        Sl = np.zeros((l,l))

        # A_p = para_dict['A']
        # B_p = para_dict['B']
        A_p = np.array([(parameters[i]%2,parameters[i+6]) for i in range(3)]) 
        B_p = np.array([(parameters[i]%2,parameters[i+6]) for i in range(3,6)]) 
        for i in range(m):
            Sm[i][(i+1) % m]=1
        for j in range(l):
            Sl[j][(j+1) % l]=1
        # print(f'SL:\n{Sl}\n,Sm:\n{Sm}')
        x = np.kron(Sl,np.eye(m))
        y = np.kron(np.eye(l),Sm)
        # print(f'x:\n{x}\n,y:\n{y}')

        A = np.zeros((l*m,l*m))
        B= np.zeros((l*m,l*m))
        for i in range(len(A_p)):
            if A_p[i][0]==0:
                A += np.linalg.matrix_power(x , (A_p[i][1] % l))
            else:
                A += np.linalg.matrix_power(y , (A_p[i][1] % l))
            if B_p[i][0]==0:
                B += np.linalg.matrix_power(x , (B_p[i][1] % m))
            else:
                B += np.linalg.matrix_power(y , (B_p[i][1] % m))
        A = A % 2
        B = B % 2
        # for i in A:
        #     for j in i:
        #         print(f'{int(j)} ',end='')
        #     print()
        # print(f'A:\n{A}\n,B:\n{B}')
        HX = np.hstack((A, B))
        HZ = np.hstack((B.T,A.T))
        # print(f'n:{2*l*m}\n,rank(A):\n{np.linalg.matrix_rank(A)},\n\nrank(B):\n{np.linalg.matrix_rank(B)}')
        css = CSSCode(HX,HZ)
        return css
    def pg_ldpc_hgp_construction(self,M):
        H1 = self.pg_ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
        H2 = self.pg_ldpc_construction(self.para_dict['p'],self.para_dict['q'],self.para_dict['m'],M)
        r1,n1 = H1.shape
        r2,n2 = H2.shape
        # HX = [H1 ⊗ In2 | Ir1 ⊗ H2.T]
        # HZ = [In1 ⊗ H2 | H1.T ⊗ Ir2]
        HX_left = np.kron(H1, np.eye(n2))
        HX_right = np.kron(np.eye(r1), H2.T)
        HX = np.hstack((HX_left, HX_right))

        HZ_left = np.kron(np.eye(n1), H2)
        HZ_right = np.kron(H1.T, np.eye(r2))
        HZ = np.hstack((HZ_left, HZ_right))

        
        return CSSCode(HX,HZ)

    def pg_ldpc_construction(self,p,q,m,M):
        '''
        M has p*q elements in the permutation group; each element is represented by a permutation from 0 to m-1
        M is a p*q * m long np.array
        '''
        H = np.zeros((m*p,m*q))
        for i in range(p*q):
            for j in range(m):
                x = int(i//q + j)
                y = int(i%q + M[m*i + j])
                H[x,y] = 1
        return H
    def quasi_cyclic_generalized_bicycle_code(self,parameters):
        if 'g' in self.para_dict.keys():
            g_size_a = self.para_dict['g']
            g_size_b = self.para_dict['g']
        else:
            g_size_a = int(self.para_dict['ga'])
            g_size_b = int(self.para_dict['gb'])
        A = parameters['A']
        B = parameters['B']

        # print(g)
        if A.shape[0] != A.shape[1]:
            raise ValueError('A should have a square shape')
        if B.shape[0] != B.shape[1]:
            raise ValueError('B should have a square shape')
        if g_size_a * A.shape[0] != g_size_b * B.shape[0]:
            raise ValueError('The shape of B(A) and B(B) is not equal.')
        A = parameters['A']
        B = parameters['B']
        B_A = self.corresponding_matrix_BA(g_size_a,A)
        B_B = self.corresponding_matrix_BA(g_size_b,B)
        # check = (B_B.T@B_A.T+B_A.T@B_B.T)%2
        # print(check.any())
        H_x = np.hstack((B_A,B_B))
        H_z = np.hstack((B_B.T,B_A.T))
        # check = (H_z @ H_x.T + B_B.T@B_A.T+B_A.T@B_B.T)%2
        # print(check.any())
        # return H_x,H_z
        return CSSCode(H_x,H_z)


    def corresponding_matrix_BA(self,g_size,A):
        r,c = A.shape[0],A.shape[1]
        for i in range(r):
            for j in range(c):
                if j==0:
                    r_matrix = self.corresponding_matrix_Ba(g_size,A[i][j])
                else:
                    r_matrix = np.hstack((r_matrix,self.corresponding_matrix_Ba(g_size,A[i][j])))
            if i==0:
                B_A = r_matrix
            else:
                B_A = np.vstack((B_A,r_matrix))
        return B_A

    def corresponding_matrix_Ba(self,g_size,a):
        B_a = np.zeros((g_size,g_size),dtype=int)
        for i in range(g_size):
            B_a = (B_a + a[i] * self.g_power(g_size,i))%2
        return B_a
    def g_power(self,g_size,p):

        g_p = np.zeros((g_size,g_size),dtype=int)
        for i in range(g_size):
            g_p[i][int((i+p)%g_size)]=1
        return g_p
    def arbitrary_bivariate_bicycle_code(self,parameters):

        # a=[a_0,a_1,...,a_{l+g-2}] is a rep of coefficient of  A=a_0*I + a_1*x + a_2*x^2 +...+ a_{l-1}*x^{l-1} + a_l*y + a_{l+1}*y^2 +...+ a_{l+g-2}*y^{g-1}
        l = self.para_dict['l']
        g = self.para_dict['g']
        a = parameters[:(l+g-1)]
        b = parameters[(l+g-1):]
        A = np.zeros((l,l,g),dtype=int)
        B = np.zeros((l,l,g),dtype=int)
        # x: g_l \otimes I_g
        for i in range(l):
            for j in range(l):
                # a_j = 1
                if a[j] == 1:
                    A[i][(i+j)%l][0]=(A[i][(i+j)%l][0]+1)%2
                if b[j] == 1:
                    B[i][(i+j)%l][0]=(B[i][(i+j)%l][0]+1)%2

        # y: I_l \otimes g_g
        for i in range(l): 
            for j in range(g-1):
                if a[j+l] == 1:
                    A[i][i][j+1]=(A[i][i][j+1]+1)%2
                if b[j+l] == 1:
                    B[i][i][j+1]=(B[i][i][j+1]+1)%2
        return self.quasi_cyclic_generalized_bicycle_code({'A':A,'B':B})
    def symmetric_quasi_cyclic_generalized_bicycle_code(self,A):
        B = A
        r = A.shape[0]
        c = A.shape[1]
        g = A.shape[2]
        for i in range(r):
            for j in range(c):
                for k in range(g):
                    B[i][j][k] = A[j][i][(g-k)%g]
        return self.quasi_cyclic_generalized_bicycle_code({'A':A,'B':B})



    



        

