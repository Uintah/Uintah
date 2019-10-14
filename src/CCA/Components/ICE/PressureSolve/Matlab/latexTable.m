function latexTable(M,Label,fileName,Caption,varargin)
%LATEXTABLE   Print a table in LaTeX tabular format.
%   LATEXTABLE(M) prints out the numeric matrix M in a LaTeX tabular
%   format. The '&' character appears between entries in a row, '\\'
%   is appended to the ends of rows, and each entry is set in math
%   mode. Complex numbers are understood, and exponentials will be
%   converted to a suitable format.
%
%   LATEXTABLE(M,'nomath') does not include the $$ needed to put each
%   entry in math mode (e.g., for use with the amsmath matrix modes).
%
%   LATEXTABLE(M,FMT) uses a format specifier FMT of the SPRINTF type for
%   each entry.
%
%   LATEXTABLE(M,FMT1,FMT2,...) works through the given format specifiers
%   on each row of M. If fewer are given than the column size of M,
%   the last is used repeatedly for the rest of the row.
%
%   S = LATEXTABLE(M,...) does not display output but returns a character
%   array S.
%
%   Examples:
%     latex( magic(4) )
%     latex( magic(4), '%i', 'nomath' )
%     latex( magic(4), '%i', '%.2f' )
%
%   See also SPRINTF, SYM/LATEX, LATEX, LATEXTABLEFACTORS.

% Revision history:
% Copyright 2002 by Toby Driscoll. Last updated 12/06/02.
% 12-JUL-2005    Oren Livne    Added comments, modified to my desired format

if (nargin < 1)
    error('Need to specify data array M');
end
if (nargin < 2)
    error('Need to specify label array Label');
end
if (nargin < 3)
    error('Need to specify an output filename (without tex extension)');
end
if (nargin < 4)
    error('Need to specify table caption');
end

fout = fopen(sprintf('%s.tex',fileName),'w');

% Print table header
fprintf(fout,'\\begin{table}[htpb]\n');
fprintf(fout,'\\centering\n');
fprintf(fout,'\\begin{tabular}{|r||');

% Print column type line
for i = 1:size(M,2)-1
    fprintf(fout,'c|');
end
fprintf(fout,'}\\hline\n');

% Print column labels line
fprintf(fout,'$%s$',Label{1});
for i = 1:size(M,2)-1
    fprintf(fout,' & $%s$',Label{i+1});
end
fprintf(fout,'\\\\ \\hline\n');

%%%%%%%%%%%%%%%%%%%%%%%% LATEX.M ORIGINAL CODE BEGIN %%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isa(M,'double')
  error('Works only for arrays of numbers.')
elseif ndims(M) > 2
  error('Works only for 2D arrays.')
end

if nargin < 5
  fmt = {'%#.5g'};
  mathstr = '$';
else
  fmt = varargin;
  idx = strmatch('nomath',fmt);
  if isempty(idx)
    mathstr = '$';
  else
    mathstr = '';
    fmt = fmt([1:idx-1 idx+1:end]);
    if isempty(fmt), fmt = {'%#.5g'}; end
  end
end

% Extend the format specifiers.
[m,n] = size(M);
if n > length(fmt)
  [fmt{end:n}] = deal(fmt{end});
end

% Create one format for a row.
rowfmt = '';
for p = 1:n
  % Remove blanks.
  thisfmt = deblank(fmt{p});

  % Add on imaginary part if needed.
  if ~isreal(M(:,p))
    % Use the same format as for the real part, but force a + sign for
    % positive numbers.
    ifmt = thisfmt;
    j = findstr(ifmt,'%');
    if ~any(strcmp(ifmt(j+1),['-';'+';' ';'#']))
      ifmt = [ifmt(1:j) '+' ifmt(j+1:end)];
    else
      ifmt(j+1) = '+';
    end
    ifmt = [ifmt 'i'];
    thisfmt = [thisfmt ifmt];
  end

  % Add to row.
  rowfmt = [rowfmt mathstr thisfmt mathstr ' & '];
end

% After last column, remove column separator and put in newline.
rowfmt(end-1:end) = [];
rowfmt = [rowfmt '\\\\\n'];

% Use it.
A = M.';
if isreal(M)
  S = sprintf(rowfmt,A);
else
  S = sprintf(rowfmt,[real(A(:)) imag(A(:))].');
end

% Remove extraneous imaginary part for real entries.
if ~isreal(M)
  zi = sprintf(ifmt,0);
  S = strrep(S,zi,blanks(length(zi)));
end

% Remove NaNs.
S = strrep(S,'$NaN$','--');
S = strrep(S,'NaN','--');

% Convert 'e' exponents to LaTeX form. This is probably really slow, but
% what can you do without regular expressions?
S = strrep(S,'e','E');
ex = min(findstr(S,'E'));
while ~isempty(ex)
  % Find first non-digit character. Where is ISDIGIT?
  j = ex+2;
  while ~isempty(str2num(S(j))) & ~strcmp(S(j),'i')
    j = j+1;
  end

  % This strips off leading '+' and zeros.
  num = sprintf('%i',str2num(S(ex+1:j-1)));

  ee = ['\times 10^{' num '}'];
  S = [S(1:ex-1) ee S(j:end)];

  ex = ex + min(findstr(S(ex+1:end),'E'));
end

% For good form, remove that last '\\'.
%S(end-2:end-1) = '  ';

% Display or output?
fprintf(fout,'%s',S);

%%%%%%%%%%%%%%%%%%%%%%%% LATEX.M ORIGINAL CODE END %%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(fout,'\\hline\n');
fprintf(fout,'\\end{tabular}\n');
fprintf(fout,'\\label{Table-%s}\n',fileName);
fprintf(fout,'\\caption{%s}\n',Caption);
fprintf(fout,'\\end{table}\n');
fclose(fout);
