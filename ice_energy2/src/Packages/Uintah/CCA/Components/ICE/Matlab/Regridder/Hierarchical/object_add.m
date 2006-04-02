function tin = object_add(tin,object,d)
%OBJECT_ADD Add object to input parameter structure.
%
%   TIN = OBJECT_ADD(TIN,OBJECT,D) adds an object OBJECT and its movement D to the list of objects
%   of the parameter structure TIN. It also synchronizes the parameters of O and D when they
%   are written to the TIN.OBJECT cell array.
%   
%   See also OBJECT_SHAPE, OBJECT_MOVEMENT, TEST_CASE.

% Author: Oren Livne
%         05/28/2004    Version 1: Created
%         06/17/2004    object,d synchronization now automatic in object_render => removed from here

%%%%% Save in the object array
tin.num_objects = tin.num_objects+1;                            % Increment number of objects
n               = tin.num_objects;                              % Alias for number of objects
tin.object{n,1} = object;                                       % Object (col 1)
tin.object{n,2} = d;                                            % Movement (col 2)
