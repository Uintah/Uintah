function efficiency = box_efficiency(points,rect)
%BOX_EFFICIENCY Box efficiency of covering flagged cells.
%   EFFICIENCY = BOX_EFFICIENCY(POINTS,RECT) returns the efficiency of the box RECT
%   of covering the flagged cells POINTS. POINTS is a binary image, and RECT is described
%   by the box convention [x1,y1,x2,y2], where (x1,y1) is the lower-left corner cell 
%   and (x2,y2) is the upper-right corner cell. RECT is assumed to be contained in
%   the scope of POINTS. If RECT is kx4, EFFICIENCY is a kx1 vector of the efficiencies
%   of each box (here efficiency = number of points in box / area of box).
% 
%   See also BOX_SIZE, BOX_VOLUME, CREATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: handles RECT kx4 arrays, not just a single box

n           = size(rect,1);
efficiency  = zeros(n,1);

for k = 1:n,
    r               = rect(k,:);                                % Box coordinates
    s               = points(r(1):r(3),r(2):r(4));              % Flag data of this box
    sz              = box_size(r);                              % Vector containing the size of the box: [size_x,size_y]
    efficiency(k)   = length(find(s))/prod(sz);                 % Percentage of flagged cells in s
end
