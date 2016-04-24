function varargout = myndgrid(varargin)
%MYNDGRID Generation of arrays for N-D functions and interpolation.
%   [X1,X2,X3,...] = MYNDGRID(x1,x2,x3,...) transforms the domain
%   specified by vectors x1,x2,x3, etc. into arrays X1,X2,X3, etc. that
%   can be used for the evaluation of functions of N variables and N-D
%   interpolation.  The i-th dimension of the output array Xi are copies
%   of elements of the vector xi.
%
%   MYNDGRID works like NDGRID, but also for 1-D cases, where it simply
%   returns X1.
%
%   See also NDGRID.

if nargin==1
    varargout = cell(1,1);
    varargout{1} = varargin{1};
    return;
end

nin = length(varargin);
nout = max(nargout,nin);
varargout = cell(1,nout);
[varargout{:}] = ndgrid(varargin{:});
