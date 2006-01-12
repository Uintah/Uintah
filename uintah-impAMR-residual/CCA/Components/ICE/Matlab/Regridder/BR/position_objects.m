function points = position_objects(tin,t)
%POSITION_OBJECTS Position objects in the binary image of flagged cells, at time t.
%
%   POINTS = POSITION_OBJECTS(TIN,T) positions the object list specified by TIN
%   (TIN includes their initial locations and time-movements), at time T.
%   The output is a binary image POINTS of flagged and non-flagged cells (objects
%   are in fact groups of flagged cells).
%   
%   See also ADD_OBJECT, TEST_CASE, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created


dim             = length(tin.domain_size);                          % Dimension of the problem
points          = zeros(tin.domain_size);                           % Empty domain
a               = cell(dim,1);                                      % Used to find the x and y coordinate (a{1} and a{2}, respectively) of an object

for n = 1:tin.num_objects,                                          % Loop over objects
    o           = tin.object{n,1};                                  % Object handle
    m           = tin.object{n,2};                                  % Movement handle
    switch (o.type)                                                 % Prepare the object
    case 'box',                                                     % Create a box        
        s           = ones(box_size(o.coords));                     % Box of the size specified by o
        offset      = size(s)/2;                                    % Offset in the center of the s-object w.r.t. the (0,0) coordinate of our domain
    case 'ball',                                                    % Create a ball (disc)
        s           = numgrid('D',o.radius);                        % A disc
        offset      = size(s)/2;                                    % Offset in the center of the s-object w.r.t. the (0,0) coordinate of our domain
    end       
    [a{:}]          = find(s);                                      % Read the x- and y- from the 2D array s (which has non-zeros integers at the flagged cells, not 1's. Not usually critical, but we will use the "+" function on binary images later on)
    
    switch (tin.object{n,2}.type)                                   % Move the object according to its movement type
    case 'line',                                                    % Linear movement
        for d = 1:dim,
            a{d}        = floor(a{d}-offset(d)+o.center(d)+t*m.direction(d));
        end
    case 'circ',                                                    % Circular movement
        a{1}        = floor(a{1}-offset(1)+m.center(1)+m.radius*cos(m.t0+t*m.dtheta));
        a{2}        = floor(a{2}-offset(2)+m.center(2)+m.radius*sin(m.t0+t*m.dtheta));
    end
    
    ind_out         = [];                                           % Remove points that are outside the domain from a
    for d = 1:dim
        ind_out     = union(ind_out,find((a{d} < 1) | (a{d} > tin.domain_size(d))));
    end
    for d = 1:dim
        a{d}(ind_out)   = [];
    end    

    if (~isempty(a{1}))
        ind             = sub2ind(tin.domain_size,a{:});                % Convert coordinates of object at time t to a 1D array
        points(ind)     = 1;                                            % Put 1's at the flagged cells
    end
end
