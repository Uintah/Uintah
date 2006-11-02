function tin = add_object(tin,o,d)
%ADD_OBJECT Add object to input parameter structure.
%
%   TIN = ADD_OBJECT(TIN,O,D) adds an object O and its movement D to the list of objects
%   of the parameter structure TIN. It also synchronizes the parameters of O and D when they
%   are written to the TIN.OBJECT cell array.
%   
%   See also CREATE_OBJECT, CREATE_MOVEMENT, TEST_CASE.

% Author: Oren Livne
%         05/28/2004    Version 1: Created

%%%%% Synchronize o, d
switch (d.type)                                                 % Correct/add object initial location according to movement type
case 'circ',                                                    % Circular movement initial theta overrides o's location
    center  = d.center + d.radius*[cos(d.t0) sin(d.t0)];          % Center of the object will be at this location
    switch (o.type)
    case 'box',                                                 % Object is a box
        b       = box_size(o.coords);                           % Box size
        o.center= [center - b/2, center + b/2];                 % Put object symmetrically around the required center
        
    case 'ball',                                                % Object is a ball
        o.center = center;                                      % Center of ball is on the circular-motion-sphere, at theta=t0
    end
end

%%%%% Save in the object array
tin.num_objects = tin.num_objects+1;                            % Increment number of objects
n               = tin.num_objects;                              % Alias for number of objects
tin.object{n,1} = o;                                            % Object (col 1)
tin.object{n,2} = d;                                            % Movement (col 2)
