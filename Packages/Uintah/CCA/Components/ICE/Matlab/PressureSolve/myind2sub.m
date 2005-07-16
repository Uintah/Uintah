function varargout = myind2sub(siz,ndx)
%MYIND2SUB Multiple subscripts from linear index.
%   MYIND2SUB is identical to IND2SUB except that it works also when
%   size(siz)=1, when it returns IND.
%
%   See also IND2SUB, SUB2IND, FIND.

nout = max(nargout,1);
varargout = cell(1,nout);
if length(siz) == 1,
    varargout{1} = ndx;
    return;
end
[varargout{:}] = ind2sub(siz,ndx);
