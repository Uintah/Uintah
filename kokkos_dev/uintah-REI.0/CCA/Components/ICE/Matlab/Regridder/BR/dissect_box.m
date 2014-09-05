function rn = dissect_box(points,r,cut,min_side)
%DISSECT_BOX Dissect a big box into two smaller boxes.
%   We we decide to dissect a box in the Berger-Rigoustos algorithm (CREATE_CLUSTER), we
%   specify the dissection cut information in the structure CUT.
%   CUT.DIM = dimension along which we try to cut (usually the longest), and CUT.PLACE is the coordinate
%   at which we dissect R. CUT.FOUND is usually set to 1, indicating that we found a place to cut R at.
%   R is cut into two smaller boxes, which are then extended to fit MIN_SIDE, the minimum side length permitted.
%   The new rectangles are output in the 2x(2d) array RN, where d=length(R)/2 is the dimension of the problem.
%   POINTS is the array of the flagged points (boolean image). We make sure that the RN boxes are first tightened
%   up around their POINTS portions, and then extended to the minimum required size.
%
%   See also CREATE_CLUSTER, COMPUTE_SIGNATURES, DISSECT_BIG_BOX, UPDATE_CLUSTER.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

dim                 = length(r)/2;                                  % Dimension of the problem
rn                  = repmat(r,[2 1]);                              % The two halves
rn(1,cut.dim+dim)   = cut.place;                                    % The "left half": change the end index in the d-direction to the cut index
rn(2,cut.dim)       = cut.place+1;                                  % The "right half": change the beginning index in the d-direction to the cut index

rn                  = box_tighten(points,rn);                       % Tight bounding box around the points
fprintf('New boxes after tightening:\n');
for h = 1:size(rn,1)
    fprintf('Half #%d coordinates [%3d,%3d,%3d,%3d]  size = %d x %d\n',h,rn(h,:),box_size(rn(h,:)));
end

rn                  = box_extend(rn,r,min_side);                    % Correct bounding boxes to at least the minimal required side length
fprintf('New boxes after extending:\n');
for h = 1:size(rn,1)
    fprintf('Half #%d coordinates [%3d,%3d,%3d,%3d]  size = %d x %d\n',h,rn(h,:),box_size(rn(h,:)));
end
