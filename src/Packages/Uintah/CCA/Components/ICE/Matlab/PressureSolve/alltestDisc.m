%function errNorm = alltestDisc(p)
%ALLTESTDISC  Test pressure equation discretization for a test case
%   battery.
%   We test pressure equation discretization error for a
%   simple 2D Poisson problem with a known solution.
%   We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. The scheme is a cell-centered,
%   finite volume, symmetric discretization on the composite AMR grid.
%   We study the discretization error vs. meshsize on a sequence of
%   increasingly finer composite grids with the same refinement "pattern".
%
%   See also: TESTDISC, TESTADAPTIVE.

% Revision history:
% 16-JUL-2005    Oren Livne    Added comments

globalParams;

initParam;                                                      % Initialize parameters structure
if (param.profile)
    profile on -detail builtin;                                 % Enable profiling
end

testCases{1} = {...
    'sinsin', ...
    'linear', ...
    'quad1', ...
    'quad2', ...
    'GaussianSource', ...
    'jump_linear', ...
    'jump_quad', ...
    'diffusion_quad_linear', ...
    'diffusion_quad_quad', ...
    };

testCases{2} = {...
    'linear', ...
    'quad1', ...
    'quad2', ...
    'sinsin', ...
    'GaussianSource', ...
    'Lshaped', ...
    'jump_linear', ...
    'jump_quad', ...
    'diffusion_quad_linear', ...
    'diffusion_quad_quad', ...
    };

testCases{3} = testCases{1};

out(0,'=========================================================================\n');
out(0,' Testing discretization accuracy on increasingly finer grids\n');
out(0,' Testing a battery of test cases\n');
out(0,'=========================================================================\n');

%=========================================================================
% Loop over test cases
%=========================================================================
p               = param;
p.verboseLevel  = 0;
for dim = 2:3
    p.dim = dim;
    out(0,'############\n');
    out(0,' %d-D tests\n',p.dim);
    out(0,'############\n');
    for count = 1:length(testCases{dim}),
        title = testCases{dim}{count};
        p.problemType           = title;
        p.outputDir             = sprintf('test_%s_%dD',title,p.dim);
        out(0,'[%3d/%3d] Running test ''%s'' ...',count,length(testCases{dim}),p.outputDir);
        [sucesss,errNorm] = testDisc(p);
        if (success == 0)
            out(0,' sucess\n');
        else
            out(0,' reached numCells = %d\n',sucess);
        end            
    end
end
if (param.profile)
    profile report;                             % Generate timing profile report
end
