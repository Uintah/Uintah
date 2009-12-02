function KK=OrderOfAccuracy

%______________________________________________________________________
%  This matlab script computes the order of accuracy

  NN = [22, 46, 64, 130, 262, 514]
  nTests = length(NN);

  L2norm   = zeros(1, nTests);
  maxError = zeros(1, nTests);

  % loop through all ofthe tests 
  % compute the L2norm and max error
  fid = fopen('OofA.dat', 'w');
  
  for n=1:nTests
    fprintf('Test: %g  NN: %g\n',n, NN(n));
    
    [L2norm(n),maxError(n), nn, NP] = amrmpm('mms', 0.1, NN(n));
    
    fprintf(fid,'%g %g %15.16E %15.16E\n',nn, NP, L2norm(n), maxError(n));
  end
  
  fclose(fid);
  close all;
  
  L2norm
  maxError

  %__________________________________
  % curve fit the L2norm error
  [coeff, fittedEq]   = fitcurve(NN, L2norm);
  [sse, FittedCurve]  = fittedEq(coeff);
  
  figure(1)
  set(1,'position',[50,100,700,700]);
  
  subplot(2,1,1),loglog(NN, L2norm,'*', NN, FittedCurve,'r')
  xlabel('Number of Nodes')
  ylabel('L2 norm (Particle Displacement)')
  grid on

  str1 = sprintf('Error = %g * NN ^{%g}',coeff(1), coeff(2));
  xpos = ( max(NN) - min(NN) )/3.0  + min(NN);
  ypos = ( max(L2norm) - min(L2norm) )/2.0 + min(L2norm);
  text(xpos, ypos, str1)
  
  %__________________________________
  % curve fit the MaxError
  [coeff2, fittedEq]  = fitcurve(NN, maxError);
  [sse, FittedCurve]  = fittedEq(coeff2);
  
  subplot(2,1,2),loglog(NN, maxError,'*', NN, FittedCurve,'r')
  ylabel('Max Error (Particle Displacement)')
  grid on

  str1 = sprintf('Error = %g * NN ^{%g}',coeff2(1), coeff2(2));
  ypos = ( max(maxError) - min(maxError) )/2.0 + min(maxError);
  text(xpos, ypos, str1)

end

%__________________________________
% This function calculates the curve fit coefficients
function [estimates, fittedEq] = fitcurve(xdata, ydata)
  
  % Call fminsearch with a random starting point.
  start_point = [1, 2];
  fittedEq = @f1;
  estimates = fminsearch(fittedEq, start_point);
  
  % f1 accepts curve parameters as inputs, and outputs sse,
  % the sum of squares error for A*X^B,
  % and the FittedCurve. FMINSEARCH only needs sse, but we want
  % to plot the FittedCurve at the end.

    function [sse, FittedCurve] = f1(params)
      A = params(1);                        
      B = params(2);                        
      FittedCurve = A .* xdata.^(B);        
      ErrorVector = FittedCurve - ydata;    
      sse = sum(ErrorVector .^ 2);          
    end
end
