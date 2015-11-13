% 1d solution for the diffusion equation

x0 = 0.0; x1 = 1.0;
D = .1;
h = .05;

% Dirichlet boundary conditions
c0 = 1;
c1 = 0;

% Grid Spacing
X = [x0:h:x1]';

% Initial Condition
C = 0*X;
Cfd = C;
Cfd(1) = c0;
Cfd(end) = c1;

% Terms used in Fourier Solution Calculation
num_fourier_terms = 1000
l0 = -(D*pi^2)/(x1^2);
b = pi/x1;
Cs = c0*(1 - X./x1);

% Finite difference stencil to be used to generate
% another solution for comparison purposes
lenX = length(X);
e = ones(lenX,1);
A = spdiags([e -2*e e], -1:1, lenX, lenX);
cfl = (D*dt)/(h^2);
disp(cfl);

max_iter = 100;
dt = .005;


CMt  = [C];
CMfd = [Cfd];
for i = 1:max_iter
  plot(X, C, 'r');
  hold on;
  plot(X, Cfd, 'b');
  hold off;
  axis([x0 x1 c1 c0]);
  drawnow;

  time = i*dt;

  % Compute Fourier solution for homogeous
  % boundary conditions
  C0 = zeros(size(X));
  for n = 1:num_fourier_terms
    l2 = l0*n^2;
    C0 = C0 + (exp(l2*time)./n)*sin(n*b.*X);
  end

  % Adjust for non-homogeous boundary conditions
  C = Cs - ((2*c0)/pi).*C0;

  % Compute finite difference solution
  Cfd = Cfd + cfl*A*Cfd;
  Cfd(1) = c0;
  Cfd(end) = c1;

  % Used for output
  CMt  = [CMt C];
  CMfd = [CMfd Cfd];
end

dlmwrite('test1_Analytic.txt',CMt');
dlmwrite('testt_fd.txt',CMfd');
