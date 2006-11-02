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

tStartCPU           = cputime;
tStartElapsed       = clock;

initParam;                                                      % Initialize parameters structure
if (param.profile)
    profile on -detail builtin;                                 % Enable profiling
end

testCases{1} = {...
    'linear', ...
    'quad1', ...
    'quad2', ...
    'sinsin', ...
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
p                           = param;
p.verboseLevel              = 0;
param.catchException        = 1;

% Write header of results section
fout = fopen(sprintf('FullResults.tex'),'w');
fprintf(fout,'%\n========================= RESULTS SECTION ===============================\n');
fclose(fout);

for dim = 1:2
    p.dim           = dim;
    p.domainSize    = repmat(1.0,[1 p.dim]);        % Domain is from [0.,0.] to [1.,1.]
    out(0,'############\n');
    out(0,' %d-D tests\n',p.dim);
    out(0,'############\n');

    % Write header of results section
    fout = fopen(sprintf('FullResults.tex'),'a');
    fprintf(fout,'%\n========================= %d-D RESULTS ===============================\n',dim);
    fprintf(fout,'\\newpage\n');
    fprintf(fout,'\\subsection{%d-D Test Cases}\n',dim);
    fprintf(fout,'\\label{Results%dD}\n',dim);
    fprintf(fout,'\n');
    if (ismember(dim,[2]))
        count = 1;
        title = testCases{dim}{count};
        p.problemType           = title;
        p.outputDir             = sprintf('test_%s_%dD',title,p.dim);
        n = 8;
        fprintf(fout,'\\subsubsection{Grid Layouts}\n');
        fprintf(fout,'\\begin{figure}[htbp]\n');
        fprintf(fout,'\\begin{center}\n');
        fprintf(fout,'\\includegraphics[width=1\\textwidth]{%s/grid%d.eps}\n',p.outputDir,n);
        fprintf(fout,'\\end{center}\n');
        fprintf(fout,'\\caption{AMR Grid Layout for $h = \\frac1%d$.} \\label{grid%d_%dD}\n',n,n,dim);
        fprintf(fout,'\\end{figure}\n');
        fprintf(fout,'\n');
    end
    fclose(fout);

    for count = 1:length(testCases{dim}),
        title = testCases{dim}{count};
        p.problemType           = title;
        p.outputDir             = sprintf('test_%s_%dD',title,p.dim);
        out(0,'[%3d/%3d] %-25s ',count,length(testCases{dim}),title);
        [errNorm,orders,success,testCPU,testElapsed] = testDisc(p);
        switch (success)
            case 0,
                result = 'failure';
            case 1,
                result = 'mem overflow';
            case 2,
                result = 'success';
            otherwise
                result = '???';
        end
        if (success > 0)
            fout = fopen(sprintf('FullResults.tex'),'a');
            fprintf(fout,'\\newpage\n');
            fprintf(fout,'\\input %s/Results\n\n',param.outputDir);
            fclose(fout);
        end
%        out(0,'%-12s  cpu=%10.2f  elapsed=%10.2f',result,testCPU,testElapsed);
        out(0,'%-12s  cpu=%10.2f  order=%4.2f',result,testCPU,orders(1));
        out(0,'\n');
    end
end
if (param.profile)
    profile report;                             % Generate timing profile report
end

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
out(0,'CPU time     = %f\n',tCPU);
out(0,'Elapsed time = %f\n',tElapsed);
