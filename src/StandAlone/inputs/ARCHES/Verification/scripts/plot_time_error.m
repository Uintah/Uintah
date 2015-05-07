close all
clear all

load FE_G1.txt
load FE_G2.txt
load FE_G3.txt
load FE_G4.txt
load RK2_G1.txt
load RK2_G2.txt
load RK2_G3.txt
load RK2_G4.txt
load RK3_G1.txt
load RK3_G2.txt
load RK3_G3.txt
load RK3_G4.txt

G = [1 .5 .25 .125];

E = size(FE_G1,1);

fprintf('There are %i equations.\n',E);
plotme = true;

while (plotme == true)
    N = input('Enter which equation # you want to plot or type 0 to exit: ');
    if (N == 0)
        plotme = false;
        fprintf('Thank you. Goodbye...\n')
    else
    %fe
    loglog(G, [FE_G1(N)/FE_G1(N) FE_G2(N)/FE_G1(N) FE_G3(N)/FE_G1(N) FE_G4(N)/FE_G1(N)], 'ro','MarkerSize',10.0)
    hold on
    loglog(G, G,'b--','LineWidth',1.0);
    
    %ssp-rk2
    loglog(G, [RK2_G1(N)/RK2_G1(N) RK2_G2(N)/RK2_G1(N) RK2_G3(N)/RK2_G1(N) RK2_G4(N)/RK2_G1(N)], 'ko','MarkerSize',10.0)
    loglog(G, G.^2,'b--','LineWidth',1.0);
    
    %ssp-rk3
    loglog(G, [RK3_G1(N)/RK3_G1(N) RK3_G2(N)/RK3_G1(N) RK3_G3(N)/RK3_G1(N) RK3_G4(N)/RK3_G1(N)], 'mo','MarkerSize',10.0)
    loglog(G, G.^4,'b--','LineWidth',1.0);
    
    grid on; 
    xlabel('\Delta t')
    ylabel('Error')
    axis([.1 1 0 1])
    set(gca, 'FontSize', 16.0);
    end
end


