function KK=OrderOfAccuracy

%______________________________________________________________________
%  This matlab script computes the order of accuracy

  NN = [22, 46, 64, 130, 262, 514]
  %NN = [22, 46, 64, 130]
  nTests    = length(NN);
  runTests  = 1;

  L2norm   = zeros(1, nTests);
  maxError = zeros(1, nTests);
  nn       = zeros(1, nTests);
  NP       = zeros(1, nTests);
  if(runTests)
    % loop through all ofthe tests 
    % compute the L2norm and max error
    fid = fopen('OofA.dat', 'w');
    fprintf(fid, '#  NN \t NP \t L2norm \t maxError \n');
    
    for n=1:nTests
      fprintf('Test: %g  NN: %g\n',n, NN(n));

      [L2norm(n),maxError(n), nn(n), NP(n)] = amrmpm('mms', 0.1, NN(n));

      fprintf(fid,'%g %g %15.16E %15.16E\n',nn(n), NP(n), L2norm(n), maxError(n));
    end

    fclose(fid);
    close all;
  end
  
  % read in the data from the file
  data = importdata('OofA.dat', ' ', 1);
  nn        = data.data(:,1)
  NP        = data.data(:,2);
  L2norm    = data.data(:,3);
  maxError  = data.data(:,4);
  %__________________________________
  % curve fit the L2norm error
  [coeff, fittedEq]   = fitcurve(nn, L2norm);
  [sse, FittedCurve]  = fittedEq(coeff);
  
  figure(1)
  set(1,'position',[50,100,700,700]);
  
  subplot(2,1,1),loglog(nn, L2norm,'*', nn, FittedCurve,'r')
  xlabel('Number of Nodes')
  ylabel('L2 norm (Particle Displacement)')
  grid on

  str1 = sprintf('Error = %g * nn ^{%g}',coeff(1), coeff(2));
  xpos = ( max(nn) - min(nn) )/3.0  + min(nn);
  ypos = ( max(L2norm) - min(L2norm) )/2.0 + min(L2norm);
  text(xpos, ypos, str1)
  
  %__________________________________
  % curve fit the MaxError
  [coeff2, fittedEq]  = fitcurve(nn, maxError);
  [sse, FittedCurve]  = fittedEq(coeff2);
  
  subplot(2,1,2),loglog(nn, maxError,'*', nn, FittedCurve,'r')
  ylabel('Max Error (Particle Displacement)')
  grid on

  str1 = sprintf('Error = %g * nn ^{%g}',coeff2(1), coeff2(2));
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
