% This routine plots the various MMS solutions.

close all
clear all

%______________ Initializtion
mms_solution = 3;  % 1 = Randy's
                   % 2 = Ann Almgren
                   % 3 = "Code verification by the MMS" SAND2000-1444, pg60
                   %      Steady state solution
          
if(mms_solution == 1)
  A  = 1;
  nu = 0.002;
  L  = 0;
  H  = 2*pi;
end
if(mms_solution == 2)
  L = 0;
  H = 1;
end

if(mms_solution == 3)
  density = 1;
  L = -pi;
  H =  pi;
end

tFinal  = 0;    % Final simulation time
N       = 50;   % resolution
dx       = (H-L)/N;  % dx
[x,y] = meshgrid(L:dx:H,L:dx:H);
 

%____________________________________
%initial condition
if(mms_solution == 1)
  u = 1 - A * cos(x).*sin(y);
  v = 1 + A * sin(x).*cos(y);
end
if(mms_solution == 2)
  u = 1 - 2*cos( 2*pi*(x) ).*sin( 2*pi*(y) );
  v = 1 + 2*sin( 2*pi*(x) ).*cos( 2*pi*(y) );
end

%dt = 0.5*h/max(max(u));
dt = 0.01;

%mov = avifile('periodicbox.avi','quality',100,'fps',5,'compression','Indeo5');
% exact solution
for t = 0:dt:tFinal
  
  %____________________________________
  %Compute the solution  
  if(mms_solution == 1)
    u = 1 - A * cos( (x-t) ).*sin( (y-t) ) * exp(-2.0 * nu * t);
    v = 1 + A * sin( (x-t) ).*cos( (y-t) ) * exp(-2.0 * nu * t);
    p = -0.25 * (A*A) * (cos(2*(x-t)) + cos(2*(y-t)) ) * exp(-4.0 * nu * t);
  end
  if(mms_solution == 2)
    u = 1 - 2*cos( 2*pi*(x - t) ).*sin( 2*pi*(y - t) );
    v = 1 + 2*sin( 2*pi*(x - t) ).*cos( 2*pi*(y - t) );
    p = -cos(4*pi*( x - t )) - cos(4*pi*( y - t ));
  end
  
  if(mms_solution == 3)
    u = exp(x).*sin(y) + exp(y).*sin(x);
    v = exp(x).*cos(y) - exp(y).*cos(x);
    p = density * ( -0.5 * ( exp(2*x) + exp(2*y) ) + exp(x + y).*cos(x + y));
  end  
  
  fprintf('Time %d Sum(divergence(u,v)) %d \n',t,sum(sum(divergence(x,y,u,v))) );
  %____________________________________
  %Plot Pressure
  figure(1)
  desc = sprintf('Time %d',t);
  subplot(2,1,1), surf(x,y,p)
  subplot(2,1,2), contourf(x,y,p)
  %colormap jet
  colorbar
  title(desc);
  %____________________________________
  %Plot velocity field
  figure(2)
  quiver(x,y,u,v,2)
  axis([L H L H]);
  pause(0.1)

  % M = getframe;
  % mov = addframe(mov,M);
end

%mov = close(mov);

