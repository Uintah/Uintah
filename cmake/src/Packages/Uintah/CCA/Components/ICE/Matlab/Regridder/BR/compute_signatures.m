function sig = compute_signatures(s)
%COMPUTE_SIGNATURES Signatures of a binary image.
%   COMPUTE_SIGNATURES(S) returns an cell array of signatures of the d-dimensional binary image s
%   (representing a rectilinear patch with flagged cells in it). sig{d} is the integral over 
%   non-zeros in S over plane cuts along dimension d.
%
%   See also CREATE_CLUSTER.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

dim         = length(size(s));                                  % Dimension of the problem
sig         = cell(dim,1);                                      % Signature in all dimensions
for d = 1:dim                                                   % Loop over dimensions
    sig{d} = s;                                                 % Start from the box
    for j = 1:dim                                               % For all dimensions...
        if (j ~= d)                                             % Loop over all dimensions except d
            sig{d} = sum(sig{d},j);                             % Sum along the j's dimension
        end
    end
end
