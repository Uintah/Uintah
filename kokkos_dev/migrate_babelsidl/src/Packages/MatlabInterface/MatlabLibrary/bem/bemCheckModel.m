function model = CheckModel(model,mode)

% FUNCTION model = CheckModel(model,mode)
%
% DESCRIPTION
% This function checks the integrity of the model matrix
% It checks whehter all fields are present, it checks and
% corrects the sizes of the fac and pts matrices and it 
% checks the triangulation for any inconsistancies.
% Although it runs a lot of checks the list is far from complete
%
% INPUT
% model       The model that needs checking
% mode        'closed' or 'open' default = 'closed'
%             Whether the model surfaces are open or closed
%
% OUTPUT
% model       In case the mistakes could be corrected, this
%             matrix contains the corrected fields
%
% CHECKS:
% - Is the surface closed ?
% - Is a node used twice in a triangle ?
% - Is every triangle oriented the same way ?
% - Has the model struct all fields necessairy ?
% - Is the surface structure complete ?
% - Is the surface CCW or CW ?
% - Do the matrices have the correct dimensions ?
% - Does the fac matrix contain valid node indices ?
%
% SOME THINGS THAT ARE NOT CHECKED (YET):
% - Do surfaces intersect ?
% - Are there surfaces with no surface area (two node coincide, but are properly triangulated) ?
%

    if nargin == 1,
        mode = 'closed';
    end    

    if ~isfield(model,'surface'),
        error('No surfaces are defined');
    end
    
    if ~iscell(model.surface),
        error('model.surface has to be a cell array');
    end
    
    for p = 1:length(model.surface),
        
        S = model.surface{p};
        
        if ~isfield(S,'node'),
            error(sprintf('No points matrix defined for surface %d',p));
        end
        
        if size(S.node,1) ~= 3,
            if size(S.node,2) == 3, 
                S.node = S.node';
                fprintf(1,'Adjusted the pts matrix in surface %d\n',p);
            end
        end    

        if ~isfield(S,'face'),
            error(sprintf('No triangulation matrix defined for surface %d',p));
        end
        
        if size(S.face,1) ~= 3,
            if size(S.face,2) == 3, 
                S.face = S.face';
                fprintf(1,'Adjusted the fac matrix in surface %d\n',p);
            end
        end    

        if min(S.face(:)) < 1,
            error('Triangulation matrix has an index smaller than one');
        end

        if max( S.face(:)) > size(S.node,2), 
            error('Triangulation matrix refers to a non-existing point'); 
        end


        fac = S.face;
        pts = S.node;
        
        ptsout = max(pts,[],2);							                % Find a point that is definitely outside the mesh
        cenpts = (pts(:,fac(1,:))+pts(:,fac(2,:))+pts(:,fac(3,:)))/3; 	% Calculate the centers of the triangles
        distance = sum((cenpts-ptsout*ones(1,size(cenpts,2))).^2,1);	% Distance to my outer point
        facindex =find(min(distance)==distance);				        % Find the triagle closest to the point that is for sure outside my mesh
        n = cross(pts(:,fac(2,facindex))-pts(:,fac(1,facindex)),pts(:,fac(2,facindex))-pts(:,fac(3,facindex)));
        s = n'*(ptsout-cenpts(:,facindex)); 				            %' If negative we need to swap this triangle
            
        if (s < 0)
            fprintf(1,'Trying to make surface %d CCW\n',p);
            S.face = S.face([1 3 2],:);
        end
        
        n = size(S.node,2);
        m = size(S.face,2);
        
        TestMatrix = zeros(n);

        for r = 1:m,
            TestMatrix(fac(1,r),fac(2,r)) = TestMatrix(fac(1,r),fac(2,r))+ 1;
            TestMatrix(fac(2,r),fac(3,r)) = TestMatrix(fac(2,r),fac(3,r))+ 1;
            TestMatrix(fac(3,r),fac(1,r)) = TestMatrix(fac(3,r),fac(1,r))+ 1;
        end

        if (max(TestMatrix(:)) > 1) | (nnz(TestMatrix-TestMatrix') > 0),
            fprintf(1,'Triangulation of surface %d is improper or has open ends\n',p);
            fprintf(1,'Trying to repair fac matrix');
            
            S.face = triCCW(S.face,S.node);

            if strcmp(mode,'closed') == 1,
                fac = S.face;
                TestMatrix = zeros(n);
    
                for r = 1:m,
                    TestMatrix(fac(1,r),fac(2,r)) = TestMatrix(fac(1,r),fac(2,r))+ 1;
                    TestMatrix(fac(2,r),fac(3,r)) = TestMatrix(fac(2,r),fac(3,r))+ 1;
                    TestMatrix(fac(3,r),fac(1,r)) = TestMatrix(fac(3,r),fac(1,r))+ 1;
                end
            
                if (max(TestMatrix(:)) > 1) | (nnz(TestMatrix-TestMatrix') > 0),
        
                    error('Repair failed, try to find the problem with the triangulation matrix');
                end
            end
        end     
        
        S.CCW = 1;
        model.surface{p} = S;
        
        if ~isfield(S,'sigma'),
           fprintf(1,'No conductivities defined for surface %d',p);
        else
            if length(S.sigma) ~= 2,
                error(sprintf('Improper conductivity matrix in surface %d',p));
            end
        end
    end
    
 return
    

