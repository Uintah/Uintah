%PLOTRESULTS  Plot ICE matlab script results versus Uintah results.
%   Problem in 1D.
%
%   We plot the state variables of the Shock Tube problem. We compare our
%   results of ICE.M with Uintah's ice results (or just plot our results
%   if compareUintah flag is off).
%
%   See also ICE, LOADUINTAH.

if (P.compareUintah)
  
  err_CC = sum(x_CC - x_ice)
  err_FC = sum(x_FC - x_FC_ice)
  %================ Plot results ================

  if ((mod(tstep,100) == 0) | (tstep == P.maxTimeSteps))
    figure(1);
    set(gcf,'position',[100,600,700,700]);

    xlo = 0.48;
    xhi = 0.52;

    subplot(2,2,1), plot(x_CC ,rho_CC, x_ice, rho_ice);
    xlim([xlo xhi]);
    legend('\rho', 'rho Uintah',2);
    grid on;

    subplot(2,2,2), plot(x_CC ,xvel_CC, x_ice, vel_ice);
    xlim([xlo xhi]);
    legend('u_1', 'u Uintah',2);
    grid on;

    subplot(2,2,3), plot(x_CC ,temp_CC, x_ice, temp_ice);
    xlim([xlo xhi]);
    legend('T', 'T Uintah',2);
    grid on;

    subplot(2,2,4), plot(x_CC ,press_CC, x_ice, press_ice);
    xlim([xlo xhi]);
    legend('p', 'p Uintah',2);
    grid on;

    figure(2);
    set(gcf,'position',[100,100,700,700]);

    subplot(3,1,1), plot(x_FC, xvel_FC, x_FC_ice, uvel_FC_ice);
    xlim([xlo xhi]);
    legend('xvel_FC','uvel_FC Uintah',2);
    grid on;

    subplot(3,1,2), plot(x_CC ,delPDilatate, x_ice, delP_ice);
    xlim([xlo xhi]);
    legend('delP','delP Uintah',2);
    grid on;

     subplot(3,1,3), plot(x_CC, press_eq_CC,'+', x_ice, press_eq_ice,'o');
    xlim([xlo xhi]);
    legend('press_eq','press_eq Uintah',2);
    grid on; 
  
    %================ Print relative errors matlab/Uintah ================
    rangeMatlab = 1:length(rho_CC);
    errorRho    = max(abs(rho_CC(rangeMatlab)           - rho_ice))      ./max(abs(rho_ice + eps));
    errorXvel   = max(abs(xvel_CC(rangeMatlab)          - vel_ice))      ./max(abs(vel_ice + eps));
    errorPress  = max(abs(press_CC(rangeMatlab)         - press_ice))    ./max(abs(press_ice + eps));
    errorTemp   = max(abs(temp_CC(rangeMatlab)          - temp_ice))     ./max(abs(temp_ice + eps));
    errordelP   = max(abs(delPDilatate(rangeMatlab)     - delP_ice))     ./max(abs(delP_ice + eps));
    errorpress_eq = max(abs(press_eq_CC(rangeMatlab)    - press_eq_ice)) ./max(abs(press_eq_ice + eps));
    erroruvel_FC  = max(abs(xvel_FC(rangeMatlab)        - uvel_FC_ice)) ./max(abs(uvel_FC_ice + eps));

    fprintf('Relative differences between Uintah and matlab results:\n');
    fprintf('press_eq: %e\n',errorpress_eq);
    fprintf('rho     : %e\n',errorRho);    
    fprintf('xvel    : %e\n',errorXvel);   
    fprintf('press   : %e\n',errorPress);  
    fprintf('temp    : %e\n',errorTemp);   
    fprintf('delP    : %e\n',errordelP);
    fprintf('vel_FC  : %e\n',erroruvel_FC);   
  %  input('hit return to continue')
  end
else
  %if (1)
  if ((mod(tstep,10) == 0) | (tstep == P.maxTimeSteps))
    %================ Plot results ================
    figure(1);
    set(gcf,'position',[100,100,900,900]);

    subplot(2,2,1), plot(x_CC,rho_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('\rho',2);
    grid on;

    subplot(2,2,2), plot(x_CC,xvel_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('u_1',2);
    grid on;

    subplot(2,2,3), plot(x_CC,temp_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('T',2);
    grid on;

    subplot(2,2,4), plot(x_CC,press_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('p',2);
    grid on;

    figure(2);
    set(gcf,'position',[300,100,900,900]);

    subplot(2,2,1), plot(x_CC,delPDilatate);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('delP',2);
    grid on;

    subplot(2,2,2), plot(x_CC,speedSound_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('speedSound',2);
    grid on;

    subplot(2,2,3), plot(x_FC,xvel_FC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('u^*',2);
    grid on;

    subplot(2,2,4), plot(x_FC,press_FC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('p^*',2);
    grid on;
  end
end

%M(tstep) = getframe(gcf);                          % Save frame for a movie
%pause
