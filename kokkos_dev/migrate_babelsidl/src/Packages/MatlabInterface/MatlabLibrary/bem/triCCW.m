function fac = triCCW(fac,pts)

% FUNCTION fac = triCCW(fac,pts)
%          surf = triCCW(surf)
%
% DESCRIPTION
% Correct a triangulation and make all triangles CCW
% 
% INPUT
% fac     Triangulation data
% pts     Points file
% 
% OUTPUT
% fac     The new triangulation file
%


    if nargin == 1,
        if isstruct(fac),
            surf = fac;
            fac = surf.fac;
            pts = surf.pts;
        else
            error('Not enough inputs');
        end
    end


    ptsout = max(pts,[],2);							% Find a point that is definitely outside the mesh
    
    cenpts = (pts(:,fac(1,:))+pts(:,fac(2,:))+pts(:,fac(3,:)))/3; 	% Calculate the centers of the triangles
    
    distance = sum((cenpts-ptsout*ones(1,size(cenpts,2))).^2,1);		% Distance to my outer point
    
    facindex =find(min(distance)==distance);				% Find the triagle closest to the point that is for sure outside my mesh
    facindex = facindex(1);
    
    
    n = cross(pts(:,fac(1,facindex))-pts(:,fac(2,facindex)),pts(:,fac(3,facindex))-pts(:,fac(2,facindex)));
    
    s = n'*(ptsout-cenpts(:,facindex)); 				%' If negative we need to swap this triangle
    
    if s < 0, temp = fac([1 3 2],facindex); fac(:,facindex) = temp; end % Correct the first one
    
    
    numfac = size(fac,2);
    lfac = [fac(1,:) fac(2,:) fac(3,:); fac(2,:) fac(3,:) fac(1,:); [1:numfac] [1:numfac] [1:numfac] ];
    
    cwok = zeros(1,numfac);
    cwok(facindex) = 1;
    lfac(3,facindex) = 0; lfac(3,facindex+numfac) = 0; lfac(3,facindex+2*numfac) = 0;	% knock out the triangles we have already checked
      
    
    while (nnz(cwok) ~= numfac),
    
        fprintf(1,'Progress: %d/%d\n',nnz(cwok),numfac);
        index = find(cwok == 1);									% find the triangles that are ok
                
        if isempty(index),
            keyboard
        end
        
        for p =index,
        
            swap = [];
            swap = [swap find((lfac(1,:) == fac(1,p))&(lfac(2,:)==fac(2,p))&(lfac(3,:) > 0))];  % find connections in the same direction
            swap = [swap find((lfac(1,:) == fac(2,p))&(lfac(2,:)==fac(3,p))&(lfac(3,:) > 0))];  % find connections in the same direction
            swap = [swap find((lfac(1,:) == fac(3,p))&(lfac(2,:)==fac(1,p))&(lfac(3,:) > 0))];  % find connections in the same direction
            
            if length(swap) > 3, error('ERROR: segment shared by more than two triangles'); end
            if length(swap) > 0, facnum = lfac(3,swap); temp = fac([1 3 2],facnum); fac(:,facnum) = temp; end
            
            facok = swap;
            facok = [facok find((lfac(1,:) == fac(2,p))&(lfac(2,:)==fac(1,p))&(lfac(3,:) > 0))];  % find connections in the opposite direction
            facok = [facok find((lfac(1,:) == fac(3,p))&(lfac(2,:)==fac(2,p))&(lfac(3,:) > 0))];  % find connections in the opposite direction
            facok = [facok find((lfac(1,:) == fac(1,p))&(lfac(2,:)==fac(3,p))&(lfac(3,:) > 0))];  % find connections in the opposite direction
                  
            facnum = lfac(3,facok);
            lfac(3,facnum) = 0; lfac(3,facnum+numfac) = 0; lfac(3,facnum+2*numfac) = 0;	% knock out the triangles that are OK
            cwok(facnum) = 1;
            cwok(p) = 2;
       end     
   end         
   
   if nargin == 1,
       surf.fac = fac;
       fac = surf;
   end
   
return     