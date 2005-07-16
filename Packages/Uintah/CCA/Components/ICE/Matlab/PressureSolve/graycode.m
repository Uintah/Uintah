function G = grayCode(N,k)
%GRAYCODE Variable-base multiple-digit gray code.
%    G = GRAYCODE(N,K) returns the gray code permutation of the integers
%    from 0 to prod(K)-1. N bust be a non-negative integer and K must be an
%    N-vector of non-negative integers of bases. K(1) is the base of the
%    right-most digit (LSB) in the N-digit string space, K(2) the base of
%    the next right digit, and so on.
%    The generated gray code is not necssarily cyclic. G is an array of size
%    prod(K)xN, whose rows are the gray-code-ordered N-digit strings.
%
%    See also: BASE2DEC, DEC2HEX, DEC2BASE, DEC2BIN.

% Revision history:
% 12-JUL-2005    Oren Livne    Created

verbose = 0;

if (nargin < 2)
    error('Must have at least two arguments');
end

if ((~isscalar(N)) & (N ~= floor(N)) || (N <= 0))
    if (N == 0)
        G = [];
        return;
    end
    error('MATLAB:graycode:FirstArg', 'N must be a non-negative integer');
end

k = k(:);
if (length(k) ~= N)
    error('MATLAB:graycode:SecondArg', 'K must be an array of length N');
end
for m = 1:N
    b = k(m);
    if ((~isscalar(b)) || (b ~= floor(b)) || (b < 0))
        error('MATLAB:graycode:SecondArg', 'K must be an array of non-negative integers');
    end
end

r = [1:N]'; %ones(N,1);

% Generate G recursively
G = [0:k(1)-1]';
if (verbose >= 1)
    fprintf('m = %d\n');
    G
end
for m = 2:N
    b       = k(m);
    s       = r(m);
    Gstar   = flipud(G);
    Gnew    = [];
    if (verbose >= 1)
        fprintf('m = %d, b = %d, s = %d\n',m,b,s);
    end
    for d = 0:b-1
        if (mod(d,2))           % d odd
            if (verbose >= 1)
                fprintf('  (G*)^(%d \\ %d)\n',d,s);
            end
            Gnew = [Gnew; insertDigit(Gstar,d,s,N)];
        else
            if (verbose >= 1)
                fprintf('  G^(%d \\ %d)\n',d,s);
            end
            Gnew = [Gnew; insertDigit(G,d,s,N)];
        end
    end
    G       = Gnew;
    if (verbose >= 1)
        G
    end
end

% Check result
fail = norm(sort(abs(diff(G,1)),2) - ...
    [repmat(0,[size(G,1)-1,size(G,2)-1]) repmat(1,[size(G,1)-1,1])]);
if (fail)
    fprintf('Gray code is incorrect!!!\n');
    %    abs(diff(G,1))
else
    if (verbose >= 1)
        fprintf('Gray code is correct.\n');
    end
end
%----------------------------------------------------------------------
function B = insertDigit(A,d,m,N)
%INSERTDIGIT Insert a digit into a set of strings.
%    B = INSERTDIGIT(A,D,M,N) follows the notation of (Gautam & Chaunhary,
%    1993) of A^(d\m). A is a set of strings of size x-1, where x > 1. If 1
%    <= m <= N-1, then B is obtained from A by inserting the digit d into
%    the position immediately to the right of digit m in every string in A.
%    If m = N, we prefix the digit d to every string in A to get B.
%
%    Examples
%        A = [3 1 2 1], N = 4. Then
%        insertDigit(A,2,2,N) returns [3 1 2 2 1]
%        insertDigit(A,2,N,N) returns [2 3 1 2 1]
%fprintf('insertDigit(A[%d x %d],d=%d,m=%d,N=%d)\n',size(A),d,m,N);
global verbose
x = size(A,2);
if (m < N)
    B = [A(:,1:x-m+1) repmat(d,[size(A,1) 1]) A(:,x-m+2:x)];
else
    B = [repmat(d,[size(A,1) 1]) A];
end
if (verbose >= 2)
    A
    B
end
