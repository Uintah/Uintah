%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function:     DQMOM Linear System Builder
%
% Author:       Charles Reid
%
% Description:  This function accepts a matrix of weights and weighted abscissas, moments, and source terms.
%               It then constructs A and B from this, and returns X.
%               X contains the source terms for the weight and weighted abscissa transport equations.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% param w   = matrix of weights' values
% param wa  = matrix of weighted abscissas' values
% param k   = matrix of moment indices
% param G   = matrix of source terms for weighted abscissas

% param condition_number  = condition number of DQMOM matrix A
% param singular          = boolean (int = 0 or 1) representing whether A is singular (1) or not (0)
% param X                 = solution vector for DQMOM linear system AX=B

function [condition_number singular X] = dqmom_linear_system(w, wa, k, G)

% diffusivity in phase space (of internal coordinate j at quad node alpha)
Gamma_xi = 0.1;

singular = 0;

N_xi = size(wa,1);
N = size(w,2);
Ntot = (N_xi+1)*N;

if (Ntot ~= size(k,1) )
  fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
  fprintf('\nError! You dont have the correct number of moments!!!\n\n');
  fprintf('You specified %0.0f moments, but you needed %0.0f moments.\n\n',size(k,1),Ntot);
  fprintf('Exiting...\n\n'); 
  fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
  return;
end

G;

for K=1:Ntot
    for alpha=1:N
        prefixA=1;
        productA=1;
        for i=1:N_xi
            prefixA=prefixA-k(K,i);
            productA=productA*(wa(i,alpha)/w(alpha))^(k(K,i));
        end
        A(K,alpha)=prefixA*productA;
    end
    
    totalsumS = 0;
    totalsumD = 0;
    for j=1:N_xi
        prefixA    = 1;
        productA   = 1;
        
        prefixS    = 1;
        productS   = 1;
        modelsumS  = 0;
        
        prefixD    = 1; %weight*diffusivity
        prefixD_1  = 1; %k_j*(k_j-1)*<xi_j>^(kj-2)
        productD_1 = 1; %prod_n(neq j) = 1 to N-xi of <xi_n>^(kn)
        prefixD_2  = 1; %k_j*k_n*<xi_j>^(kj-1)*<xi_n>^(kn-1)
        productD_2 = 1; %prod_m(neq j neq n) = 1 to N_xi of <xi_m>^(km)
        
        quadsumS = 0;
        quadsumD = 0;
        for alpha=1:N
            prefixA    = k(K,j)*(wa(j,alpha)/w(alpha))^(k(K,j)-1);
            productA   = 1;
            
            prefixS    = -k(K,j)*(wa(j,alpha)/w(alpha))^(k(K,j)-1);
            productS   = 1;
            
            prefixD    = w(alpha)*Gamma_xi;
            prefixD_1  = (k(K,j))*(k(K,j)-1)*(wa(j,alpha)/w(alpha))^(k(K,j)-2);
            productD_1 = 1;
            
            for n=1:N_xi
                if (n ~= j)
                    productA   = productA*( wa(n,alpha)/w(alpha) )^(k(K,n));
                    productS   = productS*( wa(n,alpha)/w(alpha) )^(k(K,n));
                    productD_1 = productD_1*(wa(n,alpha)/w(alpha))^(k(K,n));
                    prefixD_2  = k(K,j)*k(K,n)*((wa(j,alpha)/w(alpha))^(k(K,j)-1))*((wa(n,alpha)/w(alpha))^(k(K,n)-1));
                    
                    for m=1:N_xi
                        if (m ~= n)
                            productD_2 = productD_2*( wa(m,alpha)/w(alpha) )^(k(K,m));
                        end
                    end
                end
            end

            % model term
            modelsumS = - G(j,alpha);
            
            A(K,(j)*N+alpha)=prefixA*productA;
            
            quadsumS = quadsumS + w(alpha)*modelsumS*prefixS*productS;
            quadsumD = quadsumD + w(alpha)*Gamma_xi*(prefixD_1*productD_1 + prefixD_2*productD_2);
        end
        totalsumS = totalsumS + quadsumS;
        totalsumD = totalsumD + quadsumD;
    end
    S(K) = totalsumS;
    D(K) = totalsumD;
end

% Cmna = D_xa * wa * d(\xi_m,a)/dx_i * d(\xi_n,a)/dx_i

B = S;
B=B';

%for K=1:Ntot
%    kp = k(K,:);
%    for alpha=1:N
%        prefixA1=(1-kp(1)-kp(2));
%        productA1=(wa(1)^kp(1))*(wa(2)^kp(2));
%        A1(K,alpha)=prefixA1*productA1;
%        
%        prefixA2=(kp(1)*(wa(1)^(kp(1)-1)));
%        productA2=(wa(2)^kp(2));
%        A2(K,alpha)=prefixA2*productA2;
%        
%        prefixA3=(kp(2)*(wa(2)^(kp(2)-1)));
%        productA3=(wa(1)^kp(1));
%        A3(K,alpha)=prefixA3*productA3;
%    end
%end

if (det(A) == 0)
  fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');
  fprintf('ERROR: Your matrix is singular! Pick new moments.\n\n');
  k
  A
  B
  fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');
  singular = 1;
end

%A
%B
%X = A^-1*B
fprintf('Solving linear system...\n')
[condition_number X] = crout(A,B);

