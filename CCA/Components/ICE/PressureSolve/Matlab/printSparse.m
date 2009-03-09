function list = printSparse(A,b,fileName)
%PRINTSPARSE  Formatted sparse matrix printout.
%   LIST = PRINTSPARSE(A,B,FILENAME) prints the sparse matrix A into
%   standard output (if NARGIN = 2) or to a file FILENAME. The rows are
%   separated for reading clarity. If a non-empty right hand side B is
%   specified, B is printed for every row of A. B has to be of size MxP,
%   where A is MxN. On output, LIST = [i j aij] is the list of non-zeros in
%   A.
%
%   See also: SPARSE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

if (nargin < 3)
    f = 1;
else
    f = fopen(fileName,'w');
end

[m,n]       = size(A);
[m1,p]      = size(b);

if (~isempty(b))
    if (m1 ~= m)
        error('size(b,1) should be size(A,1)');
    end
end

[i,j,data]  = find(A);
list        = [i j data];
list        = sortrows(list);

for k = 1:size(list,1)
    if (k > 1)
        if (list(k,1) ~= list(k-1,1))
            if (~isempty(b))
                for j = 1:p
                    fprintf(f,'b(%d) = %+f\n',list(k-1,1),b(list(k-1,1),j));
                end
            end
            fprintf(f,'\n');
        end
    end
    fprintf(f,'(%5d , %5d)   %+f\n',list(k,:));
end

if (nargin >= 3)
    fclose(f);
end
