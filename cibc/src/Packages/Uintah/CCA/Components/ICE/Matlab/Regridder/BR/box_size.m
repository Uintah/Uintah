function sz = box_size(r)
%BOX_SIZE Size of a box.
%   VOL = BOX_SIZE(R) returns the box size (in all dimensions) of a d-dimensional box R, specified
%   by its lower-left and upper-right corner coordinates. If R is a kx(2d) array of
%   boxes, VOL will be a kxd array of their respective sizes.
%
%   See also BOX_INTERSECT, BOX_VOLUME, CREATE_CLUSTER..
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.
 
dim         = size(r,2)/2;                                      % Dimension of the problem
sz          = r(:,dim+1:2*dim) - r(:,1:dim) + 1;                % r = [x1_start,...,xd_start,x1_end,...xd_end]; +1 because if xi_start=xi_end, size=1
