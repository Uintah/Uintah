%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function:     LU Decomposition Solver via Crout's Method
%
% Author:       Charles Reid
%
% Description:  This function accepts a matrix A and a matrix B; it decomposes A into L and U 
%               using Crout's method, back-substitutes the result, and returns the solution X.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function X = crout(A, B)

N=3;

%A=[6,1,5;5,2,3;8,7,4]
%A=[0,0,1,1; -1,-4,2,4; -2,-16,3,12; -3,-48,4,32];
%Aorig = A;

%B = [-2;-1;-1]
%B = [3e-08; 1e-08; 2.7e-08; 6.8e-08];
%Borig = B;

for i=1:N
  AAmax = 0;
  for j=1:N
    if ( abs(A(i,j)) > AAmax )
      AAmax = abs(A(i,j));
    end
  end
  if (AAmax == 0) 
      fprintf('singular matrix!\n');
  end
  vv(i)=1/AAmax;
end

for j=1:N
  if (j > 1)
    for i=1:j-1
      sum=A(i,j);
      if (i>1)
        for k=1:i-1
          sum=sum-A(i,k)*A(k,j);
        end
        A(i,j)=sum;
%        fprintf('A(%1.0f,%1.0f) = %1.2f - line 42 \n',i,j,A(i,j));
      end
    end
  end
  
  AAmax = 0;
  for i=j:N
    sum=A(i,j);
    if (j>1)
      for k=1:j-1
        sum = sum-A(i,k)*A(k,j);
      end
      A(i,j) = sum;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - line 55 \n',i,j,A(i,j));
    end
    dum = vv(i)*abs(sum);
    if (dum >= AAmax)
      imax = i;
      AAmax = dum;
    end
  end

  if (j~=imax)
    for k=1:N
      dum = A(imax,k);
      A(imax,k)=A(j,k);
      A(j,k)=dum;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - line 69 \n',j,k,A(j,k));
    end
    vv(imax)=vv(j);
  end

  indx(j)=imax;
  if (j ~= N)
    if (A(j,j)==0)
      A(j,j)=0.0000000001;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - line 78 \n',j,j,A(j,j));
    end
    dum=1/A(j,j);
    for i=j+1:N
      A(i,j)=A(i,j)*dum;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - line 83 \n',i,j,A(i,j));
    end
  end
  
%  fprintf('finished a loop\n');
%  pause;

end
if (A(N,N)==0)
  A(N,N)=0.000000001;
end

%fprintf('A matrix:\n');
%Aorig

%fprintf('A decomposed matrix:\n');
%A

%fprintf('B matrix:\n');
%B

% forward-substitution
ii = 0;
for i=1:N
  ip = indx(i);
  sum = B(ip);
  B(ip) = B(i);
  if (ii)
    for j=ii:i-2
      sum = sum - A(i,j)*B(j);
    end
  elseif sum
    % nonzero element encountred; from now on, sum in above loop must be done
    ii = i;
  end
  B(i) = sum;
end

% back-substitution
for i=N:-1:1
  sum = B(i);

  for j=(i+1):N
    sum = sum - A(i,j)*B(j);
  end
  B(i) = sum/(A(i,i));
end

X = B;

