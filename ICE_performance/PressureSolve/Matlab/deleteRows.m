function A = deleteRows(A,rowDel,colDel,putIdentity,block)
%DELETEROWS Delete rows & cols from a sparse matrix.
%    B = DELETEROWS(A,ROWDEL,COLDEL,FLAG,BLOCK) returns a sparse matrix B of the same
%    size of A, with the rows ROWDEL and columns COLDEL overriden with
%    zeros.
%    If FLAG = 1, we also set A(ROWDEL,ROWDEL) to the identity matrix (i.e.,
%    put 1's on the diagonal). 
%    If BLOCK = 1, we delete the block A(ROWDEL,COLDEL) from A. Otherwise,
%    we delete the rows A(ROWDEL,:) and the columnds A(:,COLDEL)
%    separately from each other.
%    This is useful for deleting the part of a coarse patch underneath a fine patch. 
%    The default FLAG is 0; the default BLOCK is 0.
%
%    See also: SETPATCHINTERFACE, UPDATESYSTEM.

% Revision history:
% 15-JUL-2005    Oren Livne    Created

globalParams;

if (nargin < 4)
    putIdentity = 0;
end

if (nargin < 5)
    block = 0;
end

rowDel      = rowDel(:);
colDel      = colDel(:);
[i,j,data]  = find(A);
n           = size(A,1);
nz          = [i j data];
if (block)
    rows        = logical(ismember(i,rowDel) & ismember(j,colDel));
else
    rows        = logical(ismember(i,rowDel) | ismember(j,colDel));
end
nz(rows,:)  = [];
nz          = [nz; [n n 0]];
if (putIdentity)
    nz = [nz; [rowDel rowDel repmat(1.0,size(rowDel))]];
end
A           = spconvert(nz);
