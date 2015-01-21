function result = is_box_intersect(r,s)
%IS_BOX_INTERSECT Box intersection - boolean.
%   IS_BOX_INTERSECT(R,S) returns 1 if the boxes R and S
%   intersect, 0 if they don't. If R is a kx(2*d) collection of d-dimensional boxes, IS_BOX_INTERSECT will
%   return an a boolean array indicating the intersections of S with each R(K,:), K=1,...,size(R,1).
%
%   See also BOX_INTERSECT, SHIFT_BOX.

n   = size(r,1);
dim = length(s)/2;
t   = repmat(s,n,1);

A   = zeros(n,dim);
for d = 1:dim,
    A(:,d)   = ( ...
        ((r(:,d) <= t(:,d    )) & (t(:,d    ) <= r(:,d+dim))) | ...
        ((r(:,d) <= t(:,d+dim)) & (t(:,d+dim) <= r(:,d+dim))) ...
        );
end

result = min(A,[],2);
