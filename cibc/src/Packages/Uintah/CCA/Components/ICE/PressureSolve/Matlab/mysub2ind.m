function ndx = mysub2ind(siz,varargin)
%MYMYSUB2IND Linear index from multiple subscripts.
%   MYSUB2IND is used to determine the equivalent single index
%   corresponding to a given set of subscript values. It performs like
%   SUB2IND, but also works on 1D subscript cases.
%
%   IND = MYSUB2IND(SIZ,I) returns I if length(SIZE) = 1.
%
%   IND = MYSUB2IND(SIZ,I,J) returns the linear index equivalent to the
%   row and column subscripts in the arrays I and J for an matrix of
%   size SIZ. If any of I or J is empty, and the other is empty or scalar,
%   MYSUB2IND returns an empty matrix.
%
%   IND = MYSUB2IND(SIZ,I1,I2,...,In) returns the linear index
%   equivalent to the N subscripts in the arrays I1,I2,...,In for an
%   array of size SIZ.
%
%   Class support for inputs I,J: 
%      float: double, single
%
%   See also SUB2IND.

if length(siz)==1
    ndx = varargin{1};
    return;
end

ndx = sub2ind(siz,varargin{:});
