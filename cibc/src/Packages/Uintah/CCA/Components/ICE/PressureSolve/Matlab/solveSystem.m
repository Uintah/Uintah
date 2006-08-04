function [u,x] = solveSystem(A,b,grid,TI)
%SOLVESYSTEM  Solve the linear system of the discretization.
%   [U,X] = SOLVESYSTEM(A,B,GRID,TI) returns the solution to the system A*X = B,
%   where U is the solution vector X unwrapped into the original AMR hierarchy data (i.e.,
%   U contains the value of the solution at all patches of the GRID
%   hierarchy). TI is the transformation converting X to U.
%   We use a direct solver (MATLAB's "A\b").
%
%   See also: TESTDISC, TESTADAPTIVE.

% Revision history:
% 15-JUL-2005    Oren Livne    Created

globalParams;

out(2,'--- solveSystem() BEGIN ---\n');

tStartCPU       = cputime;
tStartElapsed   = clock;
x               = A\b;                            % Direct solver
u               = sparseToAMR(x,grid,TI,1);       % Translate the solution vector to patch-based
tCPU            = cputime - tStartCPU;
tElapsed        = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);

out(2,'--- solveSystem() END ---\n');
