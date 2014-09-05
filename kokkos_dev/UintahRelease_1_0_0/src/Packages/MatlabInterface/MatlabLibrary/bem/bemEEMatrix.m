function [EE,row,col] = bemEEMatrix(model,surf1,surf2)
% FUNCTION [EE,row,col] = bemEEMatrix(model,surf1,surf2)
%
% DEESCRIPTION
% This function computes the transfer matrix for potentials in the boundary element formulation
% Each potential can be written as a sum of contributions of normal current densities and potentials
% at each surface. This function computes the transfer between potentials
% For instance the boundary element is formulated as
%  EE*[p1 p2]' + EJ*[j1 j2]' = 0
% In this equation this function computes the EE matrix. It also computes the auto angles if they are
% present.
%
% INPUT
% model    model descriptor (struct with fields)
% surf1    the surface potential at the row space (denoted in surface numbers)
% surf2    the surface potential at the column space (denoted in surface numbers)
%
% OUTPUT
% EE       submatrix of boundary element method
% row     vector containing the surface numbers of each row
% col     vector containing the surface numbers of each column
%
% MODEL
% .surface{p}
%    .node      pts matrix [3xn]
%    .face      fac matrix [3x 2(n-2)]
%    .sigma    [cond(outside) cond(inside)]
%
% SEE ALSO bemEJMatrix, bemJEMatrix, bemJJMatrix



    % 
    % UPGRADING THE PERFORMANCE OF THE BEM
    %
    % ADDED ANOTHER FIELD IN THE MODEL STRUCTURE
    %   model.surface{}.facemaskp AND model.surface{}.facemaskj
    % THESE VECTORS CONTROL WHICH TRIANGLES ARE IGNORED IN THE COMPUTATION


    
    % COMPUTE LENGTH OF VECTORS IN ADVANCE, SHOULD BE FASTER
    
    rowlen = 0;
    for p = surf1,
        rowlen = rowlen + size(model.surface{p}.node,2);
    end
    
    row = zeros(1,rowlen);
    rows = cell(1,length(surf1));
    rowstart = 0;
    for  p = 1:length(surf1),
        rows{p} = rowstart + [1:size(model.surface{surf1(p)}.node,2)];
        row(rowstart+[1:length(rows{p})]) = surf1(p);
        rowstart = rows{p}(end);
    end    
    
    
    collen = 0;
    for p = surf2,
        collen = collen + size(model.surface{p}.node,2);
    end
    
    col = zeros(1,collen);
    cols = cell(1,length(surf2));
    colstart = 0;
    for p = 1:length(surf2),
        cols{p} = colstart + [1:size(model.surface{surf2(p)}.node,2)];
        col(colstart+[1:length(cols{p})]) = surf2(p);
        colstart = cols{p}(end);
    end    

    % PREALLOCATING MEMORY FOR FINAL MATRIX
    % SHOULD INCREASE PERFORMANCE

    EE = zeros(rowstart,colstart);

    % THIS IS A SMALL LOOP SO NOT MUCH TIME IS WASTED HERE
    
    for p = 1:length(surf1),
        for q = 1:length(surf2),
        
            
            fprintf(1,'\nCalculating potential transfer surface %d and %d\n',surf1(p),surf2(q));
            EE(rows{p},cols{q}) = CalcMatrix(model,surf1(p),surf2(q));

         end
    end        
            
return

% GEOMETRY CALCULATION SCHEMES

function EE = CalcMatrix(model,surf1,surf2)

% FUNCTION EE = CalcMatrix(model,surf1,surf2)
%
% This function deals with the analytical solutions of the various integrals in the stiffness matrix
% The function is concerned with the integrals of the linear interpolations functions over the triangles
% and takes care of the solid spherical angle. As in most cases not all values are needed the computation
% is split up in integrals from one surface to another one
%
% The computational scheme follows the analytical formulas derived by the de Munck 1992 (IEEE Trans Biomed Engng, 39-9, pp 986-90)
%
% The program is based on BEME1.m and BEME2.m (c) Stinstra 1997.
% This function has been rewritten so it does not log the data to file anymore
% and splits up the computation surface by surface.
% All comments have been translated from Dutch into English

%    fprintf(1,'Computing geometrical data : %02d %%',0);

    % GET THEM OUT OF THE CELLARRAY FOR IMPROVED PERFORMANCE

    Pts = model.surface{surf1}.node;
    Pos = model.surface{surf2}.node;
    Tri = model.surface{surf2}.face;

    if isfield(model.surface{surf2},'facmaskp'),
        facmaskp = model.surface{surf2}.facemaskp;
        Tri = Tri(:,facmaskp);
    end    

    NumPts = size(Pts,2);
    NumPos = size(Pos,2);
    NumTri = size(Tri,2);

    % Define a unitary vector 
    In = ones(1,NumTri);

    % PREALLOCATE THE MEMORY TO BE USED
    % Create an empty matrix

    GeoData = zeros(NumPts,NumTri,3);

    for p = 1:NumPts,
        
        % Print how far we are in the process, does not work in Windows
        if isunix,
%           fprintf(1,'\b\b\b\b%02d %%',floor(100*(p-1)/NumPts));
        end
            
        % Define all triangles that are no autoangles

        if surf1 == surf2,
            Sel = find((Tri(1,:) ~= p)&(Tri(2,:)~=p)&(Tri(3,:)~=p));
        else
            Sel = 1:NumTri;
        end    
        
        % Define vectors for position p

        ym = Pts(:,p)*ones(1,NumTri);
        y1 = Pos(:,Tri(1,:))-ym;
        y2 = Pos(:,Tri(2,:))-ym;
        y3 = Pos(:,Tri(3,:))-ym;

        % Found another bug and are now changing the setup of this calculation routine

        epsilon = 1e-12;		% Some accuracy parameter, used to tell whether a node lies in the plane of a triangle

        gamma = zeros(3,NumTri);

        % Speeding up the computation by splitting the formulas
        y21 = y2 - y1;
        y32 = y3 - y2;
        y13 = y1 - y3;
        Ny1 = sqrt(sum(y1.^2));
        Ny2 = sqrt(sum(y2.^2));
        Ny3 = sqrt(sum(y3.^2));
        Ny21 = sqrt(sum((y21).^2));
        Ny32 = sqrt(sum((y32).^2));
        Ny13 = sqrt(sum((y13).^2));
        
        
        
        NomGamma = Ny1.*Ny21 + sum(y1.*y21);
        DenomGamma = Ny2.*Ny21 + sum(y2.*y21);

        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(1,W) = -ones(1,size(W,2))./Ny21(W).*log(NomGamma(W)./DenomGamma(W));

        NomGamma = Ny2.*Ny32 + sum(y2.*y32);
        DenomGamma = Ny3.*Ny32 + sum(y3.*y32);

        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(2,W) = -ones(1,size(W,2))./Ny32(W).*log(NomGamma(W)./DenomGamma(W));

        NomGamma = Ny3.*Ny13 + sum(y3.*y13);
        DenomGamma = Ny1.*Ny13 + sum(y1.*y13);

        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(3,W) = -ones(1,size(W,2))./Ny13(W).*log(NomGamma(W)./DenomGamma(W));

        d = sum(y1.*cross(y2,y3));
        N = cross(y21,-y13);
        A2 = sum(N.*N);

        OmegaVec = [1 1 1]'*(gamma(3,:)-gamma(1,:)).*y1 + [1 1 1]'*(gamma(1,:)-gamma(2,:)).*y2 +[1 1 1]'*(gamma(2,:)-gamma(3,:)).*y3; %'

        % In order to avoid problems with the arctan used in de Muncks paper
        % the result is tested. A problem is that his formula under certain
        % circumstances leads to unexpected changes of signs. Hence to avoid
        % this, the denominator is checked and 2*pi is added if necessary.
        % The problem without the two pi results in the following situation in
        % which division of the triangle into three pieces results into
        % an opposite sign compared to the sperical angle of the total
        % triangle. These cases are rare but existing.
 
        Nn = (Ny1.*Ny2.*Ny3+Ny1.*sum(y2.*y3)+Ny3.*sum(y1.*y2)+Ny2.*sum(y3.*y1));

        Omega = zeros(1,NumTri);

        Vz = find(Nn(Sel) == 0);
        Vp = find(Nn(Sel) > 0); 
        Vn = find(Nn(Sel) < 0);
        if size(Vp,1) > 0, Omega(Sel(Vp)) = 2*atan(d(Sel(Vp))./Nn(Sel(Vp))); end;
        if size(Vn,1) > 0, Omega(Sel(Vn)) = 2*atan(d(Sel(Vn))./Nn(Sel(Vn)))+2*pi; end;
        if size(Vz,1) > 0, Omega(Sel(Vz)) = pi*sign(d(Sel(Vz))); end;

        zn1 = sum(cross(y2,y3).*N); 
        zn2 = sum(cross(y3,y1).*N);
        zn3 = sum(cross(y1,y2).*N);

        % Compute spherical angles
        GeoData(p,Sel,1) = In(Sel)./A2(Sel).*((zn1(Sel).*Omega(Sel)) + d(Sel).*sum((y32(:,Sel)).*OmegaVec(:,Sel))); % linear interp function corner 1
        GeoData(p,Sel,2) = In(Sel)./A2(Sel).*((zn2(Sel).*Omega(Sel)) + d(Sel).*sum((y13(:,Sel)).*OmegaVec(:,Sel))); % linear interp function corner 2
        GeoData(p,Sel,3) = In(Sel)./A2(Sel).*((zn3(Sel).*Omega(Sel)) + d(Sel).*sum((y21(:,Sel)).*OmegaVec(:,Sel))); % linear interp function corner 3

    end 
    
%    fprintf(1,'\nCalculating the potential to potential matrix 00 %%');
                    
    EE = zeros(NumPts,NumPos);
    
    % Assume every line being multiplied by this amount. 
    
    C = (1/(4*pi))*(model.surface{surf2}.sigma(1) - model.surface{surf2}.sigma(2));

    for q=1:NumPos,
    
        V = find(Tri(1,:)==q);
        for r = 1:size(V,2), EE(:,q) = EE(:,q) - C*GeoData(:,V(r),1); end;
      
        V = find(Tri(2,:)==q);
        for r = 1:size(V,2), EE(:,q) = EE(:,q) - C*GeoData(:,V(r),2); end;

        V = find(Tri(3,:)==q);
        for r = 1:size(V,2), EE(:,q) = EE(:,q) - C*GeoData(:,V(r),3); end;
        
        if isunix,
%            fprintf(1,'\b\b\b\b%02d %%',floor(100*q/NumPos));
        end    
    end
    
    if surf1 == surf2,
    
        fprintf(1,'\nCalculating diagonal elements');
        
        % added a correction for the layers outside this one.
        % It is assumed that the outer most layer has a conductivity
        % of zero. 
        
        for p = 1:NumPts,
            EE(p,p) = -sum(EE(p,:))+model.surface{surf2}.sigma(1);
        end    
         
    end        
    
    fprintf(1,'\nCompleted computation submatrix\n');
        
return

