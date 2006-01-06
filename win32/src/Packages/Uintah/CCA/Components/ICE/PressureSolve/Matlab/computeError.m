function [y,err,tau] = computeError(grid,A,b,T,TI,u,x,uExact)
%COMPUTEERROR  Compute error in discrete solution.
%   ERRNORM = COMPUTEERROR(GRID,A,B,T,TI,U,X,UEXACT) computes discretization error
%   norms of the discrete solution vs. a known exact solution. We also plot
%   the solutions and errors on each patch, if the appropriate option in
%   PARAM is specified.
%
%   See also: TESTDISC, TESTADAPTIVE.

% Revision history:
% 16-JUL-2005    Oren Livne    Created

globalParams;

out(2,'--- setupGrid() BEGIN ---\n');
tStartCPU       = cputime;
tStartElapsed   = clock;

% Plot and print discretization error at all patches
tau = sparseToAMR(b-A*AMRToSparse(uExact,grid,T,1),grid,TI,0);
f   = sparseToAMR(b,grid,TI,0);

% AMR grid norms
uValues     = sparseToAMR(x,grid,TI,1);
err         = cell(size(u));
for k = 1:grid.numLevels,
    level = grid.level{k};
    for q = 1:grid.level{k}.numPatches,
        err{k}{q} = uExact{k}{q}-uValues{k}{q};
    end
end
y       = AMRToSparse(err,grid,T,1);
err     = sparseToAMR(y,grid,TI,0);

tCPU            = cputime - tStartCPU;
tElapsed        = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);
out(2,'--- setupGrid() END ---\n');
