function r = box_tighten(points,r)

n   = size(r,1);
dim = size(r,2)/2;

for i = 1:n
    s       = points(r(i,1):r(i,3),r(i,2):r(i,4));                      % Flag data of this box
    [i1,i2] = find(s);                                                  % 2D indices of the points in this box
    r(i,:)  = [r(i,1:2) r(i,1:2)]-1 + [min(i1) min(i2) max(i1) max(i2)];% Tight bounding box around the points
end
