function reportPage(param,orders,fileName)
%REPORTPAGE  Print a results summary report page in latex format.
%   REPORTPAGE(PARAM) prints to a file a list of latex commands that decpit
%   the solution and discretization error, and show the table of error
%   norms saved in SAVERESULTS.
%
%   See also: LATEXTABLEFACTORS, SAVERESULTS.

% Revision history:
% 17-JUL-2005    Oren Livne    Added comments

n = 16;
sz = 0.32;
space = 0.5;
discErrorTitle = 'DiscError';
fout = fopen(sprintf('%s.tex',fileName),'w');

fprintf(fout,'\\subsubsection{%s}\n',param.longTitle);
fprintf(fout,'The discretization error for this problem is\n');
fprintf(fout,'$O(h^{%.1f})$ in the $L_2$ norm,',roundd(orders(1),1));
fprintf(fout,' and $O(h^{%.1f})$ in the maximum norm,',roundd(orders(2),1));
fprintf(fout,' as $h \\rightarrow 0$.\n');
fprintf(fout,'Error norms versus $h$ are shown in Table~\\ref{%s_%s}.\n',param.outputDir,discErrorTitle);
fprintf(fout,'\n');
if (ismember(param.dim,[2]))
    fprintf(fout,'\\begin{figure}[htbp]\n');
    fprintf(fout,'\\begin{center}\n');
    fprintf(fout,'\\includegraphics[width=%f\\textwidth]{%s/DiscSolution%d_L1P1.eps}\n',sz,param.outputDir,n);
    fprintf(fout,' \\hspace{%fin} ',space);
    fprintf(fout,'\\includegraphics[width=%f\\textwidth]{%s/DiscError%d_L1P1.eps}\n',sz,param.outputDir,n);
    fprintf(fout,'\\\\\n');
    fprintf(fout,'\\includegraphics[width=%f\\textwidth]{%s/DiscSolution%d_L2P1.eps}\n',sz,param.outputDir,n);
    fprintf(fout,' \\hspace{%fin} ',space);
    fprintf(fout,'\\includegraphics[width=%f\\textwidth]{%s/DiscError%d_L2P1.eps}\n',sz,param.outputDir,n);
    fprintf(fout,'\\end{center}\n');
    fprintf(fout,'\\caption{Upper left: discrete solution $u^h$ at level 1, patch 1, for\n');
    fprintf(fout,'$h = \\frac{1}{%d}$. Upper right: corresponding discretization error $U^h-u^h$.\n',n);
    fprintf(fout,'Lower left: discrete solution $u^h$ at level 2, patch 1.\n');
    fprintf(fout,'Lower right: discretization error.}\n');
    fprintf(fout,'\\label{%s_Solution}\n',param.outputDir);
    fprintf(fout,'\\end{figure}\n');
    fprintf(fout,'\n');
end
fprintf(fout,'\\input %s/%s\n',param.outputDir,discErrorTitle);

fclose(fout);
