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

function [condition_number X] = crout(A, B)

fprintf('A condition number = %g \n',cond(A));
fprintf('Inverse of condition number = %g \n',1/cond(A));

condition_number = cond(A);

N=size(A,2);

tiny = 1e-10;

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
%  fprintf('Pushing back vv[%0.0f] = 1/%0.2f = %0.4f \n',i,AAmax,1/AAmax);
end
%fprintf('\n');



% loop over columns for crout's method
for j=1:N

%  fprintf('j equals %0.0f \n',j);
  % Inner loop 1: solve for elements of U (beta_ij)
  if (j > 1)
    for i=1:j-1
%      fprintf('i equals %0.0f \n',i);
      sum=A(i,j);
%      fprintf('initializing sum as A(%0.0f,%0.0f) = %0.2f or %0.2f \n',i,j,sum,A(i,j));
      for k=1:i-1
%        fprintf('k equals %0.0f \n',k);
        sum=sum-A(i,k)*A(k,j);
%        fprintf('Sum term for k=%0.0f is %0.2f \n',k,sum);
      end
      A(i,j)=sum;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - Inner loop 1, Ln 42 \n',i,j,A(i,j));
    end
  end
  
  big = 0;
  % Inner loop 2 - solve for elements of L (alpha_ij)
  for i=j:N
    sum=A(i,j);
%    fprintf('Inner loop 2 - initializing sum as %0.2f \n',sum);
    if (j>1)
      for k=1:j-1
        sum = sum-A(i,k)*A(k,j);
%        fprintf('For k = %0.0f , sum = sum - (%g x %g) = %g \n',k,A(i,k),A(k,j),sum);
      end
      A(i,j) = sum;
%      fprintf('Inner loop 2 - j is greater than 1, and sum became %0.2f \n',sum);
%      fprintf('A(%1.0f,%1.0f) = %g - Inner loop 2, Ln 55 \n',i,j,A(i,j));
    end
    dum = vv(i)*abs(sum);
%    fprintf('Inner loop 2 - vv[i] is %0.1f; dummy is %0.2f; big is %0.2f. ',vv(i),dum,big);
    if (dum >= big)
%      fprintf('Entering imax loop now.\n');
      big = dum;
      imax = i;
%      fprintf('Inner loop 2 - setting imax = %0.0f \n',imax);
    else
%      fprintf('Did not enter imax loop.\n');
    end
  end

  % Inner loop 3 - check if you need to interchange rows (if so, do it)
  if (j~=imax)
    % yes, we do
%    fprintf('imax = %0.0f, j = %0.0f, j does NOT equal imax.\n',imax,j);
    for k=1:N
      dum = A(imax,k);
      A(imax,k)=A(j,k);
      A(j,k)=dum;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - Inner Loop 3, Ln 69 \n',j,k,A(j,k));
    end
    vv(imax)=vv(j);
  else
%    fprintf('imax = %0.0f, j = %0.0f, j equals imax.\n',imax,j);
  end

  % Inner loop 4
  indx(j)=imax;
  if (A(j,j)==0)
    A(j,j)=tiny; 
%    fprintf('A(%1.0f,%1.0f) = %1.2f - Inner loop 4, Ln 78 \n',j,j,A(j,j));
  end
  
  % Inner loop 5
  if (j ~= N)
    dum=1/A(j,j);
    for i=j+1:N
      A(i,j)=A(i,j)*dum;
%      fprintf('A(%1.0f,%1.0f) = %1.2f - Inner loop 5, Ln 83 \n',i,j,A(i,j));
    end
  end
  
%  fprintf('finished a loop\n\n');
%  pause;

end
if (A(N,N)==0)
  A(N,N)=tiny;
end

%fprintf('A matrix:\n');
%Aorig

%fprintf('A decomposed matrix:\n');
%A

%fprintf('B matrix:\n');
%B


%fprintf('\n');
%fprintf('Back-substitution routine for LU solver:\n');

% forward-substitution
%fprintf('forward substitution...\n');
ii = 0;
for i=1:N
  ip = indx(i);
%  fprintf('ip = %0.0f \n',ip);
  sum = B(ip);
%  fprintf('sum = %0.2f \n',sum);
  B(ip) = B(i);
%  fprintf('step 1 - B(%0.0f) = %0.4f \n',ip,B(ip));
  if (ii)
    for j=ii:i-1
      sum = sum - A(i,j)*B(j);
    end
  elseif sum
    % nonzero element encountred; from now on, sum in above loop must be done
    ii = i;
  end
  B(i) = sum;
%  fprintf('step 2 - B(%0.0f) = %0.4f \n',i,sum);
end

% back-substitution
%fprintf('back substitution...\n');
for i=N:-1:1
  sum = B(i);
%  fprintf('for i = %0.0f sum = %0.2f \n',i,sum);
  for j=(i+1):N
    sum = sum - A(i,j)*B(j);
%    fprintf('for j = %0.0f sum = %0.2f \n',j,sum);
  end
  B(i) = sum/(A(i,i));
%  fprintf('B(%0.0f) = %0.9f/%0.9f = %0.4f \n',i,sum,A(i,i),B(i));
end

X = B;

