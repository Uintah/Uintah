function saveResults(errNorm)
% Save errors and error factors in latex format

globalParams;

data        = errNorm(:,1);
e           = errNorm(:,2:end);
factors     = fac(e);

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
