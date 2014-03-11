function KK=OrderOfAccuracy

%______________________________________________________________________
%  This matlab script computes the order of accuracy

  Ncells = [100, 200, 400, 800]
  nTests = length(Ncells);

  Lnorm   = zeros(1, nTests);
  unix('rm OofA.dat');
  %__________________________________
  % loop through all ofthe tests 
  % compute the Lnorm
  fid = fopen('OofA.dat', 'w');
  
  for n=1:nTests
    fprintf('Test: %g  Ncells: %g\n',n, Ncells(n));
    
    [tfinal, x_CC, dx, rho_CC, vel_CC, press_CC, temp_CC]= ...
        iceTotalEnergy(Ncells(n));
        
    [Lnorm(n)] = ...
        compare_Riemann(Ncells(n), tfinal, dx, x_CC, rho_CC, vel_CC, press_CC, temp_CC)
    
    fprintf(fid,'%g %15.16E\n',Ncells(n), Lnorm(n));
  end
  
  fclose(fid);
  close all;
  
  Lnorm

  %__________________________________
  % curve fit the Lnorm error
  [coeff, fittedEq]   = fitcurve(Ncells, Lnorm);
  [sse, FittedCurve]  = fittedEq(coeff);
  
  figure(1)
  set(1,'position',[50,100,700,700]);
  
  loglog(Ncells, Lnorm,'*', Ncells, FittedCurve,'r')
  xlabel('Number of Nodes')
  ylabel('L norm (Press_CC)')
  grid on

  str1 = sprintf('Error = %g * Ncells ^{%g}',coeff(1), coeff(2));
  xpos = ( max(Ncells) - min(Ncells) )/3.0  + min(Ncells);
  ypos = ( max(Lnorm) - min(Lnorm) )/2.0 + min(Lnorm);
  text(xpos, ypos, str1)
  
  print ( 'OofA.png', '-dpng');
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
