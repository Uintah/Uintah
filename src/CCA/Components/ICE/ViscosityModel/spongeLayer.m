#! /usr/bin/octave

clear all;
close all;
format short e;
%______________________________________________________________________
%    if Fuction is inside the the sponge layer then return it's value
function  z = isInside( x, y, slMin, slMax, in)
  oneZero = (x > slMin(1)) .* (x < slMax(1)) .* (y> slMin(2)) .* (y< slMax(2));
  z = oneZero .* in;
  
  z
end

%______________________________________________________________________
%
gridMin = [0,10,0]
gridMax = [50,11,0]
nCells  = [50,2,1];
dx=(gridMax - gridMin)./nCells
gridX=gridMin(1):dx(1):gridMax(1);
gridY=gridMin(2):dx(2):gridMax(2);
gridZ=gridMin(3):dx(3):gridMax(3);
[gxx, gyy] = meshgrid( gridX,gridY);


slMin=[30,5,0]
slMax=[40,15,0]
slLength=(slMax./slMin)

isMinusFacesSmooth = [1,1,0];
isPlusFacesSmooth  = [1,1,0];

bufferMin = 2 * dx .* isMinusFacesSmooth;
bufferMax = 2 * dx .* isPlusFacesSmooth;
smoothFactorMin = [1, 1, 1] .* isMinusFacesSmooth;
smoothFactorMax = [1, 1, 1] .* isPlusFacesSmooth;

bufferMin = [0,0,0]
bufferMax = [0,0,0]

minVisc = 1;
maxVisc = 2;

fminus = 0.25 .* (1-tanh( smoothFactorMin(1) .* (slMin(1) - gxx)/slLength(1) + bufferMin(1) )) + ...
         0.25 .* (1-tanh( smoothFactorMin(2) .* (slMin(2) - gyy)/slLength(2) + bufferMin(2) ));


fplus = 0.25 .* (1-tanh(smoothFactorMax(1) .* (gxx - slMax(1))/slLength(1) + bufferMax(1) )) +...
        0.25 .* (1-tanh(smoothFactorMax(2) .* (gyy - slMax(2))/slLength(2)+ bufferMax(2) ));

        
ffminus = isInside( gxx, gyy, slMin, slMax, fminus);
ffplus  = isInside( gxx, gyy, slMin, slMax, fplus );       
       
ff = (ffminus .+ ffplus );      
#ff = (ffminus);      
visc = minVisc .+ maxVisc .* (ff);

h = figure('position',[100,100,1024,768]);

subplot(3,1,1)
mesh( gxx,gyy,fminus)

grid on
grid minor on
zlim([0,maxVisc+minVisc])

xlabel('x')
ylabel('y')
zlabel('ffminus')

subplot(3,1,2)
mesh( gxx,gyy,fplus)

grid on
grid minor on
zlim([0,maxVisc+minVisc])

xlabel('x')
ylabel('y')
zlabel('ffplus')

subplot(3,1,3)
mesh( gxx,gyy,ff)

grid on
grid minor on
zlim([0,maxVisc+minVisc])

xlabel('x')
ylabel('y')
zlabel('visc')
