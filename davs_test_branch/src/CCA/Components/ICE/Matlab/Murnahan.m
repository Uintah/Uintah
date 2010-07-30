#! /usr/bin/octave
clear all;
close all;

#__________________________________
# Octave script that plots the pressure vs rhoMicro
# for the Murnahan equation of state

n        =  7.4;     
K        =  39.0e-11;
rho0     = 1160.0;
P0       = 101325.0;
rhoMicro = [1159:0.01:1161];

#n        = 7.4;
#K        = 39.0e-11
#rho0     = 1717.0;
#P0       = 101325;
#rhoMicro = [1716:0.1:1718]


for  i=1:length(rhoMicro)
  rhoM = rhoMicro(i);
  
  if(rhoM>=rho0)                                           
    press(i)   = P0 + (1.0/(n*K))    * ( ( (rhoM/rho0) ^ n) -1.0);        
    dp_drho(i) =      (1.0/(K*rho0)) * ( (rhoM/rho0) ^ (n - 1.0) );                                                      
  else
    press(i)   = P0 * ((rhoM/rho0) ^ (1.0/(K*P0) ) );                
    dp_drho(i) = (1./(K*rho0)) * (rhoM/rho0 ^ (1.0/(K*P0) - 1.0));  
  end        
end


hold off;
xlabel ("rhoMicro");
ylabel ("Pressure");
plot(rhoMicro, press, "-;StockCode;")
grid

pause




                                              
