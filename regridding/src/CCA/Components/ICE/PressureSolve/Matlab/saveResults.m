function [errNorm,orders] = saveResults(grid,A,b,T,TI,u,uExact,expLabel,expFormat,expValue,y,err,tau,errNorm)
%SAVERESULTS  Save errors and error factors in latex format.
%   SAVERESULTS(ERRNORM) prints to a file a summary of the discretization
%   errors vs. grid resolution (or number of levels), specified by ERRNORM.
%   ERRNORM(:,1) is the grid resolution or number of levels, ERRNORM(:,2)
%   are L2 errors of all grids, and ERRNORM(:,3) the corresponding factors.
%   Similarly Columns 4 and 5 represents the maximum norm, and so on.
%
%   See also: LATEXTABLEFACTORS, TESTDISC, TESTADATPIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

% Compute error norms and label them
numNorms        = 4; %5;
normLabels      = cell(numNorms,1);
%normLabels{1}   = '\|e\|_{L_1}';
normLabels{1}   = '\|e\|_{L_2}';
normLabels{2}   = '\|e\|_{L_{\infty}}';
normLabels{3}   = '\|e\|_{H_1}';
normLabels{4}   = '\|\tau\|_{L_2}';

i1 = logical(diag(A) < 0);
i2 = logical(diag(A) >= 0);
n1 = length(full(diag(A(i1,i1))));
invA11 = spdiags(1./full(diag(A(i1,i1))),0,n1,n1);
AS = A(i2,i2) - A(i2,i1)*invA11*A(i1,i2);
ys = y(i2);

errNorm = [ errNorm; [ ...
    expValue ...
    normAMR(grid,err,'L2') ...
    normAMR(grid,err,'max') ...
    sqrt(ys'*AS*ys) ...
    normAMR(grid,tau,'L2') ...
    ]];
% out(1,'#vars = %7d  L1=%.3e  L2=%.3e  max=%.3e  H1=%.3e  tau=%.3e\n',...
%     grid.totalVars,errNorm(end,2:end));
out(1,'#vars = %7d  L2=%.3e  max=%.3e  H1=%.3e  tau=%.3e\n',...
    grid.totalVars,errNorm(end,2:end));

% Save errNorm in latex table format
data        = errNorm(:,1);
e           = errNorm(:,2:end);
factors     = fac(e);
if (size(e,1) < 2)
    factors = zeros(0,numNorms);
end
if (size(factors,1) < 2)
    orders = -ones(1,numNorms);
else
    orders = log2(factors(end,:));
end

Labels              = cell(2*numNorms+1,1);
Labels{1}           = expLabel;
fmt{1}              = expFormat;
for i = 1:size(e,2)
    data = [data e(:,i) [0; factors(:,i)]];
    Labels{2*i}     = normLabels{i};
    Labels{2*i+1}   = '{\mbox{fac.}}';
    fmt{2*i}        = '%.2e';
    fmt{2*i+1}      = '%.2f';
end

fileName            = sprintf('%s/DiscError',param.outputDir);
Caption             = sprintf('Discretization error versus base resolution $n$ for problem $%s$ ($h = 1/n$). The ``fac.'''' columns contain ratios of error norms on grids $h$ and $h/2$. The ``order'''' row is the estimated convergence order of errors as $h \\rightarrow 0$.',...
    param.problemType);
latexTableFactors(data,orders,Labels,fileName,Caption,fmt{:});

% % Plot grid
% if (grid.totalVars <= 200)
%     plotGrid(grid,sprintf('%s/grid%d.eps',param.outputDir,param.baseResolution),1,0,0,0);
% end
%  % Plot errors, solutions and save them to eps files
% if (ismember(grid.dim,[2]))
%     plotResults(grid,u,uExact,tau,param.baseResolution);
% end

% Generate report page
fileName            = sprintf('%s/Results',param.outputDir);
reportPage(param,orders,fileName);
