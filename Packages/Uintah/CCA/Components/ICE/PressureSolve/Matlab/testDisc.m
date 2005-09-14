function [errNorm,orders,success,tCPU,tElapsed,grid,A,b,x,TI] = testDisc(p)
global grid
%TESTDISC  Test pressure equation discretization.
%   We test pressure equation discretization error for a
%   simple 2D Poisson problem with a known solution.
%   We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. The scheme is a cell-centered,
%   finite volume, symmetric discretization on the composite AMR grid.
%   We study the discretization error vs. meshsize on a sequence of
%   increasingly finer composite grids with the same refinement "pattern".
%
%   See also: ADDGRIDLEVEL, ADDGRIDPATCH, TESTADAPTIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

tStartCPU           = cputime;
tStartElapsed       = clock;

if (nargin < 1)
    initParam;                                                      % Initialize parameters structure
    if (param.profile)
        profile on -detail builtin;                                 % Enable profiling
    end
else
    param = p;
end
initTest;

out(1,'=========================================================================\n');
out(1,' Testing discretization accuracy on increasingly finer grids\n');
out(1,'=========================================================================\n');

%=========================================================================
% Run discretization on a sequence of successively finer grids
%=========================================================================
[successFlag,message,messageid] = mkdir('.',param.outputDir);
errNorm                     = [];

for numCells = param.numCellsRange
    if (param.catchException)
        try;
            [errNorm,orders,grid,A,b,x,TI] = testNumCells(numCells,errNorm);
            success = 2;
        catch;
            out(1,'Failed in numCells = %d: %s\n',numCells,lasterr);
            memoverflow = findstr(lasterr,'Out of memory. Type HELP MEMORY for your options. ');
            if (isempty(memoverflow))
                success = 0;
            else
                success = 1;
            end
            break;
        end
    else
        [errNorm,orders,grid,A,b,x,TI] = testNumCells(numCells,errNorm);
        success = 2;
    end
end

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
out(1,'CPU time     = %f\n',tCPU);
out(1,'Elapsed time = %f\n',tElapsed);
if (nargin < 1)
    if (param.profile)
        profile report;                             % Generate timing profile report
    end
end

%-----------------------------------------------------------------------
function [errNorm,orders,grid,A,b,x,TI] = testNumCells(numCells,errNorm)
globalParams;
pack;
out(1,'#### nCells = %d ####\n',numCells);
param.baseResolution = numCells;

if (param.setupGrid)
    out(2,'-------------------------------------------------------------------------\n');
    out(2,' Set up grid & system\n');
    out(2,'-------------------------------------------------------------------------\n');
    out(1,'Setting up grid\n');
    grid = [];
    [grid,A,b,T,TI]     = setupGrid;
end

if (param.setupGrid & param.solveSystem)
    out(2,'-------------------------------------------------------------------------\n');
    out(2,' Solve the linear system\n');
    out(2,'-------------------------------------------------------------------------\n');
    out(1,'Solving system\n');
    [u,x] = solveSystem(A,b,grid,TI);

    out(2,'-------------------------------------------------------------------------\n');
    out(2,' Compute exact solution, error norms, plot results\n');
    out(2,'-------------------------------------------------------------------------\n');
    uExact      = exactSolutionAMR(grid,T,TI);
    surf(uExact{1}{1});
    [y,err,tau]   = computeError(grid,A,b,T,TI,u,x,uExact);
    % Save error norms to latex tables
    if (param.saveResults)
        [errNorm,orders] = saveResults(grid,A,b,T,TI,u,uExact,'n','%4d',numCells,y,err,tau,errNorm);
    end
end
