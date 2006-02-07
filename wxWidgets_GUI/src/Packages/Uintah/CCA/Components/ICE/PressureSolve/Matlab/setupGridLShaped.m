function [grid,A,b,T,TI] = setupGridLShaped(lambda)
%SETUPGRID  Set up the grid hierarchy.
%   GRID = SETUPGRID sets up the AMR grid hierarchy GRID, based on the
%   parameters of param and several hard-coded ("static") refinement
%   techniques in this function.
%
%   See also: TESTDISC, TESTADAPTIVE.

% Revision history:
% 16-JUL-2005    Oren Livne    Created

globalParams;

out(2,'--- setupGridLShaped() BEGIN ---\n');
tStartCPU       = cputime;
tStartElapsed   = clock;

A                   = [];
b                   = [];
T                   = [];
TI                  = [];
grid                = [];

grid.dim            = param.dim;
grid.domainSize     = param.domainSize;
grid.maxLevels  	= param.maxLevels;
grid.maxPatches  	= param.maxPatches;
grid.numLevels      = 0;
grid.totalVars      = 0;
grid.level          = cell(grid.maxLevels,1);

% Generate a sequence of meshsizes and radii for grids depending
% on the exchange-rate lambda
[h,R] = LShapedGrid(lambda);
if (length(h) > grid.maxLevels)
	error('Maximum number of levels exceeded\n');
end

% Add the grid levels to "grid"

for level = 1:length(h),
    % Add level
    [grid,k] = addGridLevel(grid,'meshsize',h(level));
    % One patch around middle of domain

    % Fix me!!!
    low     = 0.5-  ones(1,grid.dim);
    high    = 0.5-ones(1,grid.dim);
    
    % What does the -1 stand for?
    [grid,q] = addGridPatch(grid,k,low,high,-1);     % One global patch

    for q = 1:grid.level{k}.numPatches,
        [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
    end
end

if (param.verboseLevel >= 2)
    printGrid(grid);
end
tCPU            = cputime - tStartCPU;
tElapsed        = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);
out(2,'--- setupGridLShaped() END ---\n');
