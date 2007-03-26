%PLOTRESULTS  Plot ICE matlab script results versus Uintah results.
%   Problem in 1D.
%
%   We plot the state variables of the Shock Tube problem. We compare our
%   results of ICE.M with Uintah's ice results (or just plot our results
%   if compareUintah flag is off).
%
%   See also ICE, LOADUINTAH.

if (P.compareUintah)
    
    %================ Plot results ================
    figure(1);
    set(gcf,'position',[100,600,400,400]);
    rangeMatlab = 1:length(rho_CC)-1;
    
    subplot(2,2,1), plot(x ,rho_CC(rangeMatlab)', x, rho_ice(:,4));
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('\rho', 'rho Uintah',2);
    grid on;

    subplot(2,2,2), plot(x ,xvel_CC(rangeMatlab)', x, vel_ice(:,4));
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('u_1', 'u Uintah',2);
    grid on;

    subplot(2,2,3), plot(x ,temp_CC(rangeMatlab)', x, temp_ice(:,4));
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('T', 'T Uintah',2);
    grid on;

    subplot(2,2,4), plot(x ,press_CC(rangeMatlab)', x, press_ice(:,4));
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('p', 'p Uintah',2);
    grid on;

    figure(2);
    set(gcf,'position',[100,100,400,400]);

    subplot(2,2,1), plot(x ,delPDilatate(rangeMatlab)', x, delP_ice(:,4));
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('delP','delP Uintah',2);
    grid on;

    subplot(2,2,2), plot(x, speedSound_CC(rangeMatlab)', x, Ssound_ice(:,4));
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('speedSound','speedSound Uintah',2);
    grid on;
    
    %================ Print relative errors matlab/Uintah ================

    errorRho    = max(abs(rho_CC(rangeMatlab)'- rho_ice(:,4)))./max(abs(rho_ice(:,4) + eps));
    errorXvel   = max(abs(xvel_CC(rangeMatlab)'- vel_ice(:,4)))./max(abs(rho_ice(:,4) + eps));
    errorPress  = max(abs(press_CC(rangeMatlab)'- press_ice(:,4)))./max(abs(press_ice(:,4) + eps));
    errorTemp   = max(abs(temp_CC(rangeMatlab)'- temp_ice(:,4)))./max(abs(temp_ice(:,4) + eps));
    errordelP   = max(abs(delPDilatate(rangeMatlab)'- delP_ice(:,4)))./max(abs(delP_ice(:,4) + eps));
    errorSSound = max(abs(speedSound_CC(rangeMatlab)'- Ssound_ice(:,4)))./max(abs(Ssound_ice(:,4) + eps));
    fprintf('Relative differences between Uintah and matlab results:\n');
    fprintf('rho    : %e\n',errorRho);
    fprintf('xvel   : %e\n',errorXvel);
    fprintf('press  : %e\n',errorPress);
    fprintf('temp   : %e\n',errorTemp);
    fprintf('delP   : %e\n',errordelP);
    fprintf('SSound : %e\n',errorSSound);
    
else
    if ((mod(tstep,10) == 0) | (tstep == P.maxTimeSteps))
        %================ Plot results ================
        figure(1);
        set(gcf,'position',[100,600,400,400]);

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
        set(gcf,'position',[100,100,400,400]);

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
