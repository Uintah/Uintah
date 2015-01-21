function result = is_box_subset(r,s)
%IS_BOX_SUBSET Box inclusion.
%   IS_BOX_SUBSET(R,S) returns 1 if the box S is contained in the box R,
%   0 if not. If R is a kx(2*d) collection of d-dimensional boxes, IS_BOX_SUBSET will
%   return an a boolean array indicating whether S is a subset of each R(K,:), K=1,...,size(R,1).
%
%   See also IS_BOX_INTERSECT, SHIFT_BOX.

dim = length(s)/2;
A   = zeros(dim,1);
for d = 1:dim,
    A(d)   = ( ...
        ((r(d) <= s(d    )) & (s(d    ) <= r(d+dim))) & ...
        ((r(d) <= s(d+dim)) & (s(d+dim) <= r(d+dim))) ...
        );
end

result = min(A);
