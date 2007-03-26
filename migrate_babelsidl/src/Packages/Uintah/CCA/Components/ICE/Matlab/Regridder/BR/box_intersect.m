function A = box_intersect(r,s)
%BOX_INTERSECT Box intersection.
%   BOX_INTERSECT(R,S) returns the intersecting box of the boxes R and S
%   intersect. If R is a kx(2*d) collection of d-dimensional boxes, BOX_INTERSECT will
%   return an the intersections of S with each R(K,:), K=1,...,size(R,1).
%
%   See also BOX_SIZE, CREATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments. 

n   = size(r,1);                                                % Number of boxes in the r collection
dim = length(s)/2;                                              % Dimension of the problem
t   = repmat(s,n,1);                                            % Replicate s to the size of r
A   = zeros(size(r));                                           % The array of booleans

for d = 1:dim,                                                  % Loop over dimensions
    A(:,d)      = max(s(d),r(:,d));                             % Left coordinate of intersection
    A(:,d+dim)  = min(s(d+dim),r(:,d+dim));                     % Right coordinate of intersection
end
