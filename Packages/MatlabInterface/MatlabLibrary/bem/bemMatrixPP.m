function T = bemMatrixPP(model)
% FUNCTION T = bemMatrixPP(model)
%
% DESCRIPTION
% This function computes the transfer matrix between the inner and
% the most outer surface. It is assumed that surface 1 is the outer
% most surface and surface N the inner most. At the outermost surface
% a Neumann condition is assumed that no currents leave the model.
%
% INPUT 
% model       The model description
%
% OUTPUT
% T           The transfer matrix from inside to outside
%
% STRUCTURE DEFINITIONS
% model
%  .surface{p}           A cell array containing the different surfaces that form the model
%                        These surfaces are numbered from the outside to the inside
%      .node              A 3xN matrix that describes all node positions
%      .face              A 3xM matrix that describes which nodes from one triangle
%                        Each trianlge is represented by one column of this matrix
%      .sigma            The conductivity inside and outside of this boundary
%                        The first element is the conductivity outside and the second the one inside
%      .cal              The vector describes the calibration of the potential the nodes in this vector
%                        will be summed to zero in the deflation process.
%
% NOTE The fields mentioned here are the fields the program uses for its computation purposes. However
%      more fields may be present that resulted from other programs, like channels files etc. These fields
%      will be ignored by the current program
%
% STABILITY
% The program is still in a testing phase and some features have not been tested thoroughly. Please report
% any bugs you encounter.
%
% FUTURE WORK
% - Test/expand the capabilities for doing more than two surfaces
% - Add options to switch on or off the radon integration and support full analytical
%   as well as radon integral solutions
% - Need to add some more features on computing the auto solid angles. At the moment the notion of an eigen value 0
%    is used to computed these auto solid angles (angle computed from a point on the triangle itself).
% - Upgrade some old code that computes the current density using the bem method rather than taking a numerical gradient
%

    % VALIDATE THE INTEGRITY OF THE MODEL BY DOING SOME SIMPLE TESTS ON THE GEOMETRY
    % - IS THE BOUNDARY CLOSED ?
    % - ARE ALL TRIANGLES ORIENTED THE SAME WAY ?
    % - IS A NODE BEING USED MORE THAN ONCE IN A TRIANGLE ?

    model = bemCheckModel(model);

    % hsurf = heart surface
    % bsurf = body surface (one or more surfaces)
    
    hsurf = length(model.surface);
    bsurf = 1:(hsurf-1);
    
    % The EE Matrix computed the potential to potential matrix.
    % The boundary element method is formulated as
    %
    % EE*u + EJ*j = source
    %
    % Here u is the potential at the surfaces and j is the current density normal to the surfac.
    % and source are dipolar sources in the interior of the mode space, being ignored in this formulation
    
    
    [EE,row] = bemEEMatrix(model,[bsurf hsurf],[bsurf hsurf]);
    Gbh = bemEJMatrix(model,bsurf,hsurf);
    Ghh = bemEJMatrix(model,hsurf,hsurf);
    
    % Do matrix deflation
    
    % The matrix deflation is performed on the potential to potential
    % matrix. I still have to include a more throrough deflation for
    % the case that there are multiple surfaces as proposed by Lynn and Timlake
    
    % There are two options :
    % If cal is defined for each surface use, those calibration matrices
    % and make the sum of the matrices they indicate zero
    % Or else use every node with equal weighing
    
    test = 1;
    for p = 1:length(model.surface),
        test = test * isfield(model.surface{p},'cal');
    end
    
    if test == 1,
        % constuct a deflation vector
        % based on the information of which 
        % nodes sum to zero.
        
        eig = ones(size(EE,2),1);
        p = zeros(1,size(EE,2));
        k = 0;
        for q = 1:length(model.surface),
            p(model.surface{q}.cal+k) = 1;
            k = k + size(model.surface{q}.node,2);
        end
        p = p/nnz(p);
        EE = EE + eig*p;        
    else
        EE = EE + 1/(size(EE,2))*ones(size(EE));
    end
   
    % Get the proper indices for column and row numbers
    
    b = find(row ~= hsurf);   % body surface indices
    h = find(row == hsurf);   % heart surface indices
    
    Pbb = EE(b,b);
    Phh = EE(h,h);
    Pbh = EE(b,h);
    Phb = EE(h,b);
    
    iGhh = inv(Ghh);
    
    % Formula as used by Barr et al.
    
    % The transfer function from innersurface to outer surfaces (forward problem)
    T = inv(Pbb - Gbh*iGhh*Phb)*(Gbh*iGhh*Phh-Pbh);
    
return    
