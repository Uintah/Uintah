function phi = ssprk3( fname, phi0, t )

nt       = length(t);  % number of time locations
[neq,nc] = size(phi0); % neq = number of equations.

if( nt<2 ) error('must specify at least two points for time.'); end
if( nc>1 ) error('initial guess must be a column vector'); end

% build an array to hold the solution at each time
phi = zeros(neq,nt);
phi(:,1) = phi0;

for i=2:nt
   
   % determine the time step
   dt = t(i)-t(i-1);
   
   % calculate the RK coefficients
   u1 = phi0 + dt * feval(fname,t(i),phi0);
   u2 = 3/4*phi0 + 1/4*u1 + 1/4*dt*feval(fname,t(i),u1);
   phi(:,i) = 1/3*phi0 + 2/3*u2 + 2/3*dt*feval(fname,t(i),u2);
   
   % reset the guess for the next solution
   phi0 = phi(:,i);

   %fprintf('%1.4f %1.4f %1.4f\n',u1,u2,phi0);

end
