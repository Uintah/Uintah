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

function X = dqmom_linear_system(w, wa, k, G)

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
    
    totalsumB=0;
    for j=1:N_xi
        prefixA = 1;
        productA=1;
        productB=1;
        modelsumB=0;
        for alpha=1:N
            prefixA=(k(K,j))*(wa(j,alpha)/w(alpha))^(k(K,j)-1);
            productA=1;
            productB=w(alpha);
            productB=productB*(-k(K,j)*( wa(j,alpha)/w(alpha) )^(k(K,j)-1));
            for n=1:N_xi
                if (n ~= j)
                    productA = productA*( wa(n,alpha)/w(alpha) )^(k(K,n));
                    productB = productB*( wa(n,alpha)/w(alpha) )^(k(K,n));
                end
            end
            modelsumB = modelsumB - G(j,alpha);
            A(K,(j)*N+alpha)=prefixA*productA;
        end
        totalsumB = totalsumB + productB*modelsumB;
    end
    B(K) = totalsumB;
end

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
  fprintf('Exiting...\n\n');
  fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');
  return;
end

A
B
X = A^-1*B

