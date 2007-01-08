function [EJ,row,col] = bemEJMatrix(model,surf1,surf2,mode)
% FUNCTION [EJ,row,col] = bemEJMatrix(model,surf1,surf2,mode)
%
% DEESCRIPTION
% This function computes the transfer matrix for potentials in the boundary element formulation
% from the normal current density.
% Each potential can be written as a sum of contributions of normal current densities and potentials
% at each surface. This function computes the transfer between potentials
% For instance the boundary element is formulated as
%  EE*[p1 p2]' + EJ*[j1 j2]' = 0
% In this equation this function computes the EJ matrix. 
%
% INPUT
% model    model descriptor (struct with fields)
% surf1    the surface potential at the row space (denoted in surface numbers)
% surf2    the surface potential at the column space (denoted in surface numbers)
% mode     'lin' or 'const' : The interpolation function at a triangle
%
% OUTPUT
% EJ       submatrix of boundary element method
% row      vector containing the surface number for each row
% col      vector containing the surface number for each column
%
%
% SEE ALSO bemEEMatrix, bemJEMatrix, bemJJMatrix

    if nargin == 3,
        mode = 'lin';
    end
    

    rowstart = 0;
    row =[];
    for p = 1:length(surf1),
        rows{p} = rowstart + [1:size(model.surface{surf1(p)}.node,2)];
        rowstart = rows{p}(end);
        row = [ row surf1(p)*ones(size(rows{p}))];
    end    

    colstart = 0;
    col = [];
    for p = 1:length(surf2),
        switch mode
        case 'lin',
            cols{p} = colstart + [1:size(model.surface{surf2(p)}.node,2)];
        case 'const'
            cols{p} = colstart + [1:size(model.surface{surf2(p)}.face,2)];
        end    
        colstart = cols{p}(end);
        col = [col surf2(p)*ones(size(cols{p}))];
    end    

    EJ = zeros(rowstart,colstart);
    
    for p = 1:length(surf1),
        for q = 1:length(surf2),
        
            fprintf(1,'Calculating current density to potential transfer surface %d and %d\n',surf1(p),surf2(q));
            EJ(rows{p},cols{q}) = CalcMatrix(model,surf1(p),surf2(q),mode);
            fprintf(1,'\n');
         end
    end        
   
    
return

% GEOMETRY CALCULATION SCHEMES

function EJ = CalcMatrix(model,surf1,surf2,mode)

% FUNCTION EJ = CalcMatrix(model,surf1,surf2)
%
% This function deals with the analytical solutions of the various integrals in the stiffness matrix
% The function is concerned with the integrals of the linear interpolations functions over the triangles
% and takes care of the solid spherical angle. As in most cases not all values are needed the computation
% is split up in integrals from one surface to another one
%
% The computational scheme follows the analytical formulas derived by the de Munck 1992 (IEEE Trans Biomed Engng, 39-9, pp 986-90)
%
% The program is based on SBEM_SSOURCE.m (c) Stinstra 1997.
% All comments have been translated from Dutch into English

%    fprintf(1,'Computing geometrical data : %02d %%',0);

    Pts = model.surface{surf1}.node;
    Pos = model.surface{surf2}.node;
    Tri = model.surface{surf2}.face;

    NumPts = size(Pts,2);
    NumPos = size(Pos,2);
    NumTri = size(Tri,2);

    if strcmp(mode,'lin') == 1 ,
               
        if isfield(model.surface{surf2},'facmaskj'),
            facmaskj = model.surface{surf2}.facemaskj;
            Tri = Tri(:,facmaskj);
        end        
               
        EJ = zeros(NumPts,NumPos);
        for k = 1:NumPts,
            W = bem_radon(Tri',Pos',Pts(:,k)');
            for l = 1:NumTri,
                EJ(k,Tri(:,l)) = EJ(k,Tri(:,l)) + (1/(4*pi))*W(l,:);
            end
            if isunix,
%               fprintf(1,'\b\b\b\b%02d %%',floor(100*k/NumPts));
           end               
        end
    end
   
    if strcmp(mode,'const') == 1,
            
        for p = 1:NumPts,
            if isunix,
%                fprintf(1,'\b\b\b\b%02d %%',floor(100*(p-1)/NumPts));
            end
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

            NomGamma = sqrt(sum(y1.*y1).*sum((y2-y1).*(y2-y1))) + sum(y1.*(y2-y1));
            DenomGamma = sqrt(sum(y2.*y2).*sum((y2-y1).*(y2-y1))) + sum(y2.*(y2-y1));

            W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
            gamma(1,W) = -ones(1,size(W,2))./sqrt(sum((y2(:,W)-y1(:,W)).*(y2(:,W)-y1(:,W)))).*log(NomGamma(W)./DenomGamma(W));

            NomGamma = sqrt(sum(y2.*y2).*sum((y3-y2).*(y3-y2))) + sum(y2.*(y3-y2));
            DenomGamma = sqrt(sum(y3.*y3).*sum((y3-y2).*(y3-y2))) + sum(y3.*(y3-y2));

            W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
            gamma(2,W) = -ones(1,size(W,2))./sqrt(sum((y3(:,W)-y2(:,W)).*(y3(:,W)-y2(:,W)))).*log(NomGamma(W)./DenomGamma(W));
    
            NomGamma = sqrt(sum(y3.*y3).*sum((y1-y3).*(y1-y3))) + sum(y3.*(y1-y3));
            DenomGamma = sqrt(sum(y1.*y1).*sum((y1-y3).*(y1-y3))) + sum(y1.*(y1-y3));

            W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
            gamma(3,W) = -ones(1,size(W,2))./sqrt(sum((y1(:,W)-y3(:,W)).*(y1(:,W)-y3(:,W)))).*log(NomGamma(W)./DenomGamma(W));

            d = sum(y1.*cross(y2,y3));
            N = cross((y2-y1),(y3-y1));

            OmegaVec = [1 1 1]'*(gamma(3,:)-gamma(1,:)).*y1 + [1 1 1]'*(gamma(1,:)-gamma(2,:)).*y2 +[1 1 1]'*(gamma(2,:)-gamma(3,:)).*y3; %'

            % In order to avoid problems with the arctan used in de Muncks paper
            % the result is tested. A problem is that his formula under certain
            % circumstances leads to unexpected changes of signs. Hence to avoid
            % this, the denominator is checked and 2*pi is added if necessary.
            % The problem without the two pi results in the following situation in
            % which division of the triangle into three pieces results into
            % an opposite sign compared to the sperical angle of the total
            % triangle. These cases are rare but existing.
 
            Nn = (sqrt(sum(y1.*y1)).*sqrt(sum(y2.*y2)).*sqrt(sum(y3.*y3))+sqrt(sum(y1.*y1)).*sum(y2.*y3)+sqrt(sum(y3.*y3)).*sum(y1.*y2)+sqrt(sum(y2.*y2)).*sum(y3.*y1));

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
        
	        C = ([1 1 1]'*sum(y1.*n)).*n;

	        Temp = (-1/(4*pi))*(sum(cross(y1,y2).*n).*gamma(1,:) + sum(cross(y2,y3).*n).*gamma(2,:) + sum(cross(y3,y1).*n).*gamma(3,:) - sum(n.*C).*Omega);

	        EJ(p,:) = Temp;
        end
    end 

return
    
function W = bem_radon(TRI,POS,OBPOS);
% W = bem_radon(TRI,POS,OBPOS);
%
% Radon integration over plane triangle, for mono layer 
% (linear distributed source strenghth)
% When the distance between the OBPOS and POS is smaller than
% eps this singularity is analytically handled by bem_sing 
%
% INPUT
%  TRI... Indexes of triangles
%  POS... Positions [x,y,z] of triangles
%  OBPOS... Position of observation point [x,y,z]
%
% OUTPUT
%  W... Weight, linearly distributed over triangle.
%       W(10,3) is weight of triangle 10, vertex 1
%

% J. Radon (1948), Zur Mechanischen Kubatur,
% Monat. Mathematik, 52, pp 286-300

% initial weights
s15 = sqrt(15);
w1 = 9/40;
w2 = (155+s15)/1200; w3 = w2; w4 = w2;
w5 = (155-s15)/1200; w6 = w5; w7 = w6;
s  = (1-s15)/7;
r  = (1+s15)/7;

% how many, allocate output
nrTRI = size(TRI,1);
nrPOS = size(POS,1);
W = zeros(nrTRI,3);

% move all positions to OBPOS as origin
POS = POS - ones(nrPOS,1)*OBPOS;
I = find(sum(POS'.^2)<eps); Ising = [];
if ~isempty(I), 
    Ising  = [];
    for p = 1:length(I);
        [tx,dummy] = find(TRI==I(p)); 
        Ising = [Ising tx'];
    end
end


% corners, center and area of each triangle
P1 = POS(TRI(:,1),:); 
P2 = POS(TRI(:,2),:); 
P3 = POS(TRI(:,3),:);
C = (P1 + P2 + P3) / 3;
N = cross(P2-P1,P3-P1);
A = 0.5 * sqrt(sum(N'.^2))';

% point of summation (positions)
q1 = C;
q2 = s*P1 + (1-s)*C;
q3 = s*P2 + (1-s)*C;
q4 = s*P3 + (1-s)*C;
q5 = r*P1 + (1-r)*C;
q6 = r*P2 + (1-r)*C;
q7 = r*P3 + (1-r)*C;

% norm of the positions
nq1 = sqrt(sum(q1'.^2))';
nq2 = sqrt(sum(q2'.^2))';
nq3 = sqrt(sum(q3'.^2))';
nq4 = sqrt(sum(q4'.^2))';
nq5 = sqrt(sum(q5'.^2))';
nq6 = sqrt(sum(q6'.^2))';
nq7 = sqrt(sum(q7'.^2))';

% weight factors for linear distribution of strengths
a1 = 2/3; b1 = 1/3;
a2 = 1-(2*s+1)/3; b2 = (1-s)/3;
a3 = (s+2)/3; b3 = (1-s)/3;
a4 = (s+2)/3; b4 = (2*s+1)/3;
a5 = 1-(2*r+1)/3; b5 = (1-r)/3;
a6 = (r+2)/3; b6 = (1-r)/3;
a7 = (r+2)/3; b7 = (2*r+1)/3;

% calculated different weights
W(:,1) = A.*((1-a1)*w1./nq1 + (1-a2)*w2./nq2 + (1-a3)*w3./nq3 + (1-a4)*w4./nq4 + (1-a5)*w5./nq5 + (1-a6)*w6./nq6 + (1-a7)*w7./nq7);
W(:,2) = A.*((a1-b1)*w1./nq1 + (a2-b2)*w2./nq2 + (a3-b3)*w3./nq3 + (a4-b4)*w4./nq4 + (a5-b5)*w5./nq5 + (a6-b6)*w6./nq6 + (a7-b7)*w7./nq7);
W(:,3) = A.*(b1*w1./nq1 + b2*w2./nq2 + b3*w3./nq3 + b4*w4./nq4 + b5*w5./nq5 + b6*w6./nq6 + b7*w7./nq7);

% do singular triangles!
for i=1:length(Ising),
	I = Ising(i);
	W(I,:) = bem_sing(POS(TRI(I,:),:));
end    
    
return    

function W = bem_sing(TRIPOS);
% W = bem_sing(TRIPOS);
%
% W(J) is the contribution at vertex 1 from unit strength
% at vertex J, J = 1,2,3

% find point of singularity and arrange tripos
ISIN = find(sum(TRIPOS'.^2)<eps);
if isempty(ISIN), error('Did not find singularity!'); return; end
temp = [1 2 3;2 3 1;3 1 2];
ARRANGE = temp(ISIN,:);

% Divide vertices in RA .. RC
% The singular node is called A, its cyclic neighbours B and C
RA = TRIPOS(ARRANGE(1),:);
RB = TRIPOS(ARRANGE(2),:);
RC = TRIPOS(ARRANGE(3),:); 

% Find projection of vertex A (observation point) on the line
% running from B to C
[RL,AP] = laline(RA,RB,RC);

% find length of vectors BC,BP,CP,AB,AC
BC = norm(RC-RB);
BP = abs(RL)*BC;
CP = abs(1-RL)*BC;
AB = norm(RB-RA);
AC = norm(RC-RA);

% set up basic weights of the rectangular triangle APB
% WAPB(J) is contribution at vertex A (== observation position!) 
% from unit strength in vertex J, J = A,B,C
if abs(RL) > eps,
	a = AP; 
	b = BP;
	c = AB;
	log_term = log( (b+c)/a );
	WAPB(1) = a/2 * log_term;
	w = 1-RL; 
	WAPB(2) = a* (( a-c)*(-1+w) + b*w*log_term )/(2*b);
	w = RL;
	WAPB(3) = a*w *( a-c  +  b*log_term )/(2*b);
else
	WAPB = [0 0 0];
end

% set up basic weights of the rectangular triangle APB
% WAPC(J) is contribution at vertex A (== observation position!) 
% from unit strength in vertex J, J = A,B,C
if abs(RL-1) > eps,
	a = AP;
	b = CP;
	c = AC;
	log_term = log( (b+c)/a );
	WAPC(1) = a/2 * log_term;
	w = 1-RL;
	WAPC(2) = a*w *( a-c  +  b*log_term )/(2*b);
	w = RL;
	WAPC(3) = a* (( a-c)*(-1+w) + b*w*log_term )/(2*b);
else
	WAPC = [ 0 0 0];
end

% Compute total weights taking into account the position P on BC
if RL<0, WAPB = -WAPB; end
if RL>1, WAPC = -WAPC; end
W = WAPB + WAPC;

% arrange back
W(ARRANGE) = W;

return
%%%%% end mono_ana %%%%%


%%%%% local function %%%%%%
function [rl,ap] = laline(ra,rb,rc);
% find projection P of vertex A (observation point) on the line
% running from B to C
% rl = factor of vector BC, position p = rl*(rc-rb)+rb
% ap = distance from A to P
%
% called by mono_ana (see above)

% difference vectors
rba = ra - rb;
rbc = rc - rb;

% normalize rbc
nrbc = rbc / norm(rbc) ;

% inproduct and correct for normalization
rl = rba * nrbc';
rl = rl / norm(rbc) ;

% point of projection
p = rl*(rbc)+rb;

% distance
ap = norm(ra-p);

%%%%% end laline %%%%%
return
