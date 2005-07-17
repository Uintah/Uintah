function errNorm = saveResults(grid,A,b,T,TI,expLabel,expFormat,expValue,y,err,tau,errNorm)
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
numNorms        = 5;
normLabels      = cell(numNorms,1);
normLabels{1}   = '\|e\|_{L_1}';
normLabels{2}   = '\|e\|_{L_2}';
normLabels{3}   = '\|e\|_{L_{\infty}}';
normLabels{4}   = '\|e\|_{H_1}';
normLabels{5}   = '\|\tau\|_{L_2}';

i1 = logical(diag(A) < 0);
i2 = logical(diag(A) >= 0);
n1 = length(full(diag(A(i1,i1))));
invA11 = spdiags(1./full(diag(A(i1,i1))),0,n1,n1);
AS = A(i2,i2) - A(i2,i1)*invA11*A(i1,i2);
ys = y(i2);

errNorm = [ errNorm; [ ...
    expValue ...
    normAMR(grid,err,'L1') ...
    normAMR(grid,err,'L2') ...
    normAMR(grid,err,'max') ...
    sqrt(ys'*AS*ys) ...
    normAMR(grid,tau,'L2') ...
    ]];
out(1,'#vars = %7d  L1=%.3e  L2=%.3e  max=%.3e  H1=%.3e  tau=%.3e\n',...
    grid.totalVars,errNorm(end,2:end));

% Save errNorm in latex table format
data        = errNorm(:,1);
e           = errNorm(:,2:end);
factors     = fac(e);
if (size(e,1) < 2)
    factors = zeros(0,size(e,2));
end

Labels              = cell(2*numNorms+1,1);
Labels{1}           = expLabel;
fmt{1}              = expFormat;
for i = 1:size(e,2)
    data = [data e(:,i) [0; factors(:,i)]];
    Labels{2*i}     = normLabels{i};
    Label{2*i+1}    = '\|e\|_{L_{\infty}}';
    fmt{2*i}        = '%.3e';
    fmt{2*i+1}      = '%.3f';   
end

fileName            = sprintf('%s/DiscError',param.outputDir);
Caption             = sprintf('Discretization error');
%data
latexTableFactors(data,Label,fileName,Caption,fmt{:});
