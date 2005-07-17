function errNorm = reportPage(param)
%REPORTPAGE  Save errors and error factors in latex format.
%   REPORTPAGE(ERRNORM) prints to a file a summary of the discretization
%   errors vs. grid resolution (or number of levels), specified by ERRNORM.
%   ERRNORM(:,1) is the grid resolution or number of levels, ERRNORM(:,2)
%   are L2 errors of all grids, and ERRNORM(:,3) the corresponding factors.
%   Similarly Columns 4 and 5 represents the maximum norm, and so on.
%
%   See also: LATEXTABLEFACTORS, TESTDISC, TESTADATPIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments
fprintf(f,'\\subsubsection{%s}\n',param.longTitle);
fprintf(f,'The discretization error for this problem is\n');
fprintf(f,'$O(h^{%.1f})$ in the $L_2$ norm');
fprintf(f,' and $O(h^{%.1f})$ in the maximum norm,');
fprintf(f,' as $h \\rightarrow 0$.\n');
fprintf(f,'Error norms are shown in Table.~\\ref{%s};\n');
fprintf(f,'\n');
fprintf(f,'\\begin{figure}[htbp]\n');
fprintf(f,'\\begin{center}\n');
fprintf(f,'\\includegraphics[width=0.45\\textwidth]{ProblemA_1Level/DiscSolution%d_L1P1.eps}\n');
fprintf(f,'\\includegraphics[width=0.45\\textwidth]{ProblemA_1Level/DiscError%d_L1P1.eps}\n');
fprintf(f,'\\end{center}\n');
fprintf(f,'\\caption{Left: discrete solution $u^h$ at level 1, patch 1, for\n');
fprintf(f,'$h = \\frac{1}{32}$. Right: discretization error $U^h-u^h$.}\n');
fprintf(f,'\\label{solution32}\n');
fprintf(f,'\\end{figure}\n');
fprintf(f,'\n');
fprintf(f,'\\input %s/DiscError\n',);
