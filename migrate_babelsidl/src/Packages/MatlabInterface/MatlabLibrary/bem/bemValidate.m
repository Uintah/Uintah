function bemValidate
% FUNCTION bemValidate
%
% DESCRIPTION
% This function generates some boundary element models (spherical) and computes
% both the boundary element method and the analytical solution in order to establish
% whether updates to the code still do their job
%
% INPUT -
%
% OUTPUT -
%


    % Model I   
    % radius 1 and 2    
    % conductivity 1

    
    model = bemGenerateSphere([2 1],[0 1 0],0.13);

    model = bemCheckModel(model);
    Transfer = bemMatrixPP(model);
    
    % Generate input and output potential distributions
    Uh = anaSolveSphere('U',model.surface{2}.pts,[0 0 0],[1 0 1],[1 2],[1 1 0]);
    Ub = anaSolveSphere('U',model.surface{1}.pts,[0 0 0],[1 0 1],[1 2],[1 1 0]);
 
    Uforward = Transfer*Uh;
    
    % Compare
    
    f1 = figure('name','original');
    bemPlotSurface(model.surface{1},Ub,'blue','colorbar');
    
    f2 = figure('name','forward BEM'); 
    bemPlotSurface(model.surface{1},Uforward,'blue','colorbar');
    
    rdm = errRDM(Ub,Uforward)
    mag = errMAG(Ub,Uforward)