function saveResults(errNorm)
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

data        = errNorm(:,1);
e           = errNorm(:,2:end);
factors     = fac(e);
if (size(e,1) < 2)
    factors = zeros(0,size(e,2));
end

fmt{1}      = '%4d';
for i = 1:size(e,2)
    data = [data e(:,i) [0; factors(:,i)]];
    fmt{2*i} = '%.3e';
    fmt{2*i+1} = '%.3f';
end

Label       = cell(9,1);
Label{1}    = 'n';
Label{2}    = '\|e\|_{L_2}';
Label{3}    = '{\mbox{factor}}';
Label{4}    = '\|e\|_{L_{\infty}}';
Label{5}    = '{\mbox{factor}}';
Label{6}    = '\|e\|_{H_1}';
Label{7}    = '{\mbox{factor}}';
Label{8}    = '\|e\|_{H_1,max}';
Label{9}    = '{\mbox{factor}}';

fileName    = sprintf('%s/DiscError',param.outputDir);
Caption     = sprintf('Discretization error');

latexTableFactors(data,Label,fileName,Caption,fmt{:});
