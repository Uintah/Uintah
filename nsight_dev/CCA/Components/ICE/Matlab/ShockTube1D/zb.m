function zb
%ZB Zha-Bilgen flux vector splitting method for the shocktube problem.
%
%   This script is a MATLAB translation of the zhabilg FORTRAN code
%   implementing the Zha-Bilgen first order flux vector splitting for the
%   compressible Euler equations.
%   Small amount of constant-coefficient artificial viscosity added to
%   help stabilize solution on some problems.
%
%   References:
%   [1] http://capella.colorado.edu/~laney/ch18soft/zb.f
%   [2] Laney, C. B., Computational Gasdynamics, Cambridge University
%   Press, Cambridge, 1998, p. 379.
%
%   See also ICE.

%--- REVISION INFORMATION ---
% 25-APR-04 (Oren Livne) Created

N = 50;     % Number of grid points
q = zeros(N+3,3);           % State vector at cell centers
f = zeros(N+2,3);           % Numerical flux at cell faces
me = zeros(N+1,1);          % energy momentum
av = zeros(N+3,3);          % Artificial viscosity

% Set parameter values
P.gamma = 1.4;
P.R = 287.0;
P.cflfac = 0.4;
P.cu = P.R/(P.gamma-1.0);
P.cp = P.gamma*P.R/(P.gamma-1.0);

% Read input parameters from file
ff = fopen('input.dat','r');
aa = fscanf(ff,'%f',1);
bb = fscanf(ff,'%f',1);
t = fscanf(ff,'%f',1);
pl = fscanf(ff,'%f',1);
rhol = fscanf(ff,'%f',1);
ul = fscanf(ff,'%f',1);
pr = fscanf(ff,'%f',1);
rhor = fscanf(ff,'%f',1);
ur = fscanf(ff,'%f',1);
fclose(ff);

% Initialize useful quantities and print them out
al = sqrt(P.gamma*pl/rhol);
ar = sqrt(P.gamma*pr/rhor);
delta_x = (bb-aa)/real(N);
me(1) = max([abs(ul+al),abs(ur+ar),abs(ul-al),abs(ur-ar)]);
delta_t = P.cflfac*delta_x/me(1);
itert = floor(t/delta_t);
lambda = delta_t/delta_x;
cfl = lambda*me(1);
fprintf('Final time requested = %f\n', t);
fprintf('Delta_t = %f\n', delta_t);
fprintf('Delta_x = %f\n', delta_x);
fprintf('Lambda = %f\n', lambda);
fprintf('Initial CFL number = %f\n', cfl);
fprintf('# iterations = %d\n', itert);

% Convert primitive variables to conservative variables.
rul = rhol*ul;
rur = rhor*ur;
retl = .5*rul*ul + pl/(P.gamma-1.);
retr = .5*rur*ur + pr/(P.gamma-1.);
fprintf('retl = %f\n', retl);
fprintf('retr = %f\n', retr);

% Construct the initial conditions for the Riemann problem.
i = [0:N+2]';
x = aa + (bb-aa)*(i-1)/(1.0*N);
iNegative = find(x < 0);
iPositive = find(x >= 0);
q(iNegative,:) = repmat([rhol rul retl], [length(iNegative),1]);
q(iPositive,:) = repmat([rhor rur retr], [length(iPositive),1]);

t = 0;

% Main loop.
for j = 1:itert

    t = t + delta_t;
    cfl = 0.;

    % Find conserative numerical fluxes.
    for i = 1:N+2
        f(i,:) = zbflux(q(i,:),q(i+1,:));
        av(i,:) = 0.;
        %          if(i.gt.0) av1(i) = .03*(u1(i+1)-2.*u1(i)+u1(i-1))
        %          if(i.gt.0) av2(i) = .03*(u2(i+1)-2.*u2(i)+u2(i-1))
        %          if(i.gt.0) av3(i) = .03*(u3(i+1)-2.*u3(i)+u3(i-1))
    end % for i = 1:N+2

    % Update conserved variables.
    for i = 2:N+2
        if (j > 2)
            q(i,:) = q(i,:) - lambda*(f(i,:)-f(i-1,:));
        else
            q(i,:) = q(i,:) - lambda*(f(i,:)-f(i-1,:)) + av(i,:);
        end

        if ((q(i,1) < 0) | (q(i,3) < 0))
            fprintf('WARNING: Negative density or energy\n');
            fprintf('# time steps = %d\n', j);
            fprintf('Grid point = %d\n', i);
            fprintf('x = %f\n', x(i));
            fprintf('Density = %f\n', q(i,1));
            fprintf('Total energy per unit volume = %f\n', q(i,3));
            error('Exiting due to unphysical values');
        end
        u = q(i,2)/q(i,1);
        p = (P.gamma-1.)*(q(i,3) - .5*q(i,2)*q(i,2)/q(i,1));
        if (p < 0)
            fprintf('WARNING: Negative pressure\n');
            fprintf('# time steps = %d\n', j);
            fprintf('Grid point = %d\n', i);
            fprintf('x = %f\n', x(i));
            fprintf('Pressure = %f\n', p);
            error('Exiting due to unphysical values');
        end
        a = sqrt(P.gamma*p/q(i,1));
        me(i) = max(abs(u+a),abs(u-a));
        cfl = max(cfl,lambda*me(i));
    end % i=1:N+1

    if (cfl > 1.0)
        fprintf('WARNING: CFL condition violated\n');
        fprintf('# time steps = %d\n', j);
        fprintf('Maximum CFL number = %f\n', cfl);
    end

end % for j = 1:itert

fprintf('Calculation complete.\n');
fprintf('Final time = %f\n', t);
fprintf('Final CFL number = %f\n' , cfl);

% Write output files
ff = fopen('zb_matlab.out','w');
u = zeros(N+3,1);
p = zeros(N+3,1);
a = zeros(N+3,1);

for i = 2:N+2
    u(i) = q(i,2)/q(i,1);
    p(i) = (P.gamma-1.)*(q(i,3) - .5*q(i,2)*q(i,2)/q(i,1));
    a(i) = sqrt(P.gamma*p(i)/q(i,1));
    fprintf(ff,'%.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f\n',...
        x(i),p(i),u(i),a(i),q(i,1),P.cu*log(p(i))-P.cp*log(q(i,1)),u(i)/a(i),q(i,2),lambda*me(i));
end % for i = 1:N+1
fclose(ff);

figure(1);
clf;

subplot(2,2,1);
plot(x,p,'bo-');
temp = axis;
axis([x(2) x(end-1) min(p) max(p)]);
grid on;
xlabel('x');
ylabel('pressure (N/m^2)');

subplot(2,2,2);
plot(x,u,'bo-');
temp = axis;
axis([x(2) x(end-1) min(u) max(u)]);
grid on;
xlabel('x');
ylabel('Velocity (m/s)');

subplot(2,2,3);
plot(x,q(:,1),'bo-');
temp = axis;
axis([x(2) x(end-1) min(q(:,1)) max(q(:,1))]);
grid on;
xlabel('x');
ylabel('Density (kg/m^3)');

subplot(2,2,4);
plot(x,a,'bo-');
temp = axis;
axis([x(2) x(end-1) min(a(2:end-1)) max(a(2:end-1))]);
grid on;
xlabel('x');
ylabel('Speed of Sound (m/s');

%------------------------------------------------------------------------
function f = zbflux(ql,qr)
% ZBFLUX Zha-Bilgen flux vector splitting
P.gamma = 1.4;

rl = ql(1);
rul = ql(2);
retl = ql(3);
rr = qr(1);
rur = qr(2);
retr = qr(3);

% Convert conservative variables to primitive variables.
rhol = rl;
rhor = rr;
ul = rul/rhol;
ur = rur/rhor;
pl = (P.gamma-1.)*(retl - .5*rul*rul/rhol);
pr = (P.gamma-1.)*(retr - .5*rur*rur/rhor);
hl = (retl+pl)/rhol;
hr = (retr+pr)/rhor;
al = sqrt(P.gamma*pl/rhol);
ar = sqrt(P.gamma*pr/rhor);
Ml = ul/al;
Mr = ur/ar;

% Compute positive splitting of p.
if (Ml <= -1.0)
    pp = 0.;
    pap = 0.;
elseif (Ml < 1.0)
    pp = .5*(1.+Ml)*pl;
    pap = .5*(ul+al)*pl;
else
    pp = pl;
    pap = pl*ul;
end

% Compute negative splitting of M and p.
if (Mr <= -1.0)
    pm = pr;
    pam = pr*ur;
elseif(Mr <= 1.0)
    pm = .5*(1.-Mr)*pr;
    pam =.5*(ur-ar)*pr;
else
    pm = 0.;
    pam = 0.;
end

% Compute conserative numerical fluxes.
%     f1 = .5*(rul+rur)-.5*rhor*abs(ur) +.5*rhol*abs(ul)
%     f2 = .5*(rul*ul+rur*ur+pl+pr)-.5*rur*abs(ur)+.5*rul*abs(ul)
%    *     -.5*pr*Mr +.5*pl*Ml
%     f3 = .5*(rhor*hr*ur+rhol*hl*ul) -.5*retr*abs(ur)
%    *     +.5*retl*abs(ul)-.5*pr*ar + .5*pl*al
f = [ ...
    max(0.,ul)*rhol + min(0.,ur)*rhor; ...
    max(0.,ul)*rul  + min(0.,ur)*rur + pp + pm; ...
    max(0.,ul)*retl + min(0.,ur)*retr + pap + pam; ...
    ];
