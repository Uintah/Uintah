function Handle = bemPlotSurface(varargin)
% FUNCTION Handle = bemPlotSurface([Figure],Surface,U,options,...)
%
% DESCRIPTION
% This function plots a triangulated surface and projects the surface potential
% on top of it.
%
% INPUT
% Figure           Figure handle in which to plot the data (default is the current figure)
% Surface          The surface to plot
% U                The potential data of the surface (a vector containing the potential for all nodes)
% options          Additional options see OPTIONS
%
% OUTPUT
% Handle           A vector to the handle of all graphicss objects used in the plot
%
% OPTIONS
% Possible options, formatted as 'option',value1,value2 and so on
% These options can be added to the end of the input
%
% The following options are available 
%  'limit',Origin,Normal   - Limits the display to half space defined by a plane
%                            Origin and normal define the plane that is used for cutting the model
%  'contour',Contour       - Plot contour lines at value specified in Contour
%                            Contour is a vector with the potentials for each ispotential line.
%                            e.g. Contour = 1:5 will generate five isopotential lines at a value of 1,2,3,4 and 5
%  'blue'                  - PPT style graph used to make the output suitable for powerpoint. A blue background
%                            and no axis
%  'alpha',alphavalue      - Make the surface transparant using the alphasettings of opengl
%  'colorbar'              - Add an colorbar to the image
%
% SURFACE STRUCTURE
%  surface               A structured matrix containing the surface descriptor
%      .pts              A 3xN matrix that describes all node positions
%      .fac              A 3xM matrix that describes which nodes from one triangle
%                        Each trianlge is represented by one column of this matrix



% read figure handle
k =1;
if ishandle(varargin{1}),
   Figure = varargin{1};
   k = k + 1;
else
   Figure = gcf;
end

% A remainder of some older code
BlackWhite = [];

% read Pos Tri Boundary and U;
surface = varargin{k};
k = k + 1;
Pos = surface.pts;
Tri = surface.fac;
Boundary = ones(1,size(Pos,2)); % default boundary

    
U = Boundary;  % Just give all surfaces an equal data set

if (nargin >= k),
    if ~ischar(varargin{k}),
        U = varargin{k};
        k = k + 1;
        if size(U,1) > 1, U = U'; end
    end
end

if length(U) < size(Pos,2),
    U(length(U)+1:size(Pos,2)) = NaN;
end    

Iso = 0;
Limit = 0;
Normal = [0 0 1];
Origin = [0 0 0];
blue = 0;
alpha = 0;
Bound = [1:max(Boundary)]; % all boundaries
cb = 0;

while(k <= nargin)
   if ~ischar(varargin{k}),
      error('unknown argument');
   end
   switch varargin{k},
   case 'limit'
      Limit =1;
      Origin = varargin{k+1};
      Normal = varargin{k+2};
      k=k+3;
   case 'boundary'
      Bound = varargin{k+1};
      k=k+2;
   case 'contour'
      Iso = 1;
      Contour = varargin{k+1};
      k=k+2;
   case 'blue'
      blue = 1;   
      k = k + 1;
   case 'alpha'
      alpha = varargin{k+1};
      k = k + 2; 
  case 'colorbar'
      cb = 1;
      k = k + 1;
   otherwise
      error('option unknown');
   end
end

% end read args
%make figure active

figure(Figure)

if ~isempty(BlackWhite),
   G = gray(40);
   colormap(G(10:40,:));
else
   colormap jet(15);
end

for p = Bound,
   I = zeros(1,size(Tri,2));
	V = find(Boundary == p);
	for q = 1:size(V,2),
		I1 = find((Tri(1,:) == V(q))|(Tri(2,:) == V(q))|(Tri(3,:) == V(q)));
		I(I1) = ones(1,size(I1,2));
	end
	bTri = Tri([1:3],find(I));
   if Limit ==1,
      Handle = PSGenTriLim(Pos,bTri,U,Origin,Normal);
      if Iso == 1,
         warning('Contour mode and limit mode cannot used simulteneously');
         PlotIsoLines(Pos,bTri,U,Contour);
      end
   else
      Handle = PSGenTri(Pos,bTri,U);
      if Iso==1,
         PlotIsoLines(Pos,bTri,U,Contour);
      end
   end
end

axis equal;
axis vis3d;

if alpha > 0,
    L = findall(Handle,'type','patch');
    set(L,'FaceAlpha',alpha);
else   
    set(gcf,'renderer','opengl')
end

if cb == 1,
    colorbar;
end

if blue == 1,
    % Set Background to blue and text to white/yellow
    set(gcf,'color',[0 0 0.4]);
    Text = findall(allchild(gcf),'type','text');
    for p = 1:length(Text), set(Text(p),'color',[1 1 0.6],'FontSize',18); end
    Text = findall(allchild(gcf),'type','axes');
    for p = 1:length(Text), set(Text(p),'XColor',[1 1 0.6],'YColor',[1 1 0.6],'ZColor',[1 1 0.6],'Color',[0.2 0.2 0.6],'FontSize',18); end

    set(gcf,'InvertHardcopy','off');
    axis off
end

return

function Handle = PSGenTri(Pos,Tri,U)

Handle = [];
hold on;

S = 1:size(Tri,2);
for k = 1:size(S,2),
	X = Pos(1,Tri(:,S(k)));
	Y = Pos(2,Tri(:,S(k)));
   Z = Pos(3,Tri(:,S(k)));
   V = U(Tri(:,S(k)));
      
   Handle(end+1) = patch(X,Y,Z,V','EdgeColor',[0.4 0.4 0.4]);
end
return

function PlotIsoLines(Pos,Tri,U,Contour)

S = 1:size(Tri,2);
NumS = size(S,2);

N = zeros(size(Pos));
n = cross(Pos(:,Tri(1,:))-Pos(:,Tri(2,:)),Pos(:,Tri(2,:))-Pos(:,Tri(3,:)));

for p = 1:NumS,
   N(:,Tri(:,p)) = N(:,Tri(:,p)) + n(:,[p p p]);
end

Nlen = sqrt(sum(N.^2));
N = N./([1 1 1]'*Nlen);
xlen = 0.0025*(max(Pos(:))-min(Pos(:)));

Pos = Pos + xlen*N;

hold on;
for p = 1:NumS,
   
   q = S(p);
	Q = Tri(:,q);

	[Value,I] = sort(U(Q));
	MiN = I(1); Mi = Value(1);
	MidN = I(2); Mid = Value(2);
	MaN = I(3); Ma = Value(3);

	Con = find((Contour < Ma)&(Contour > Mi));
	NCon = size(Con,2);

	if size(Con,1) > 0,
		for p = Con(1):Con(NCon),

			if Contour(p) > U(Q(MidN)),
 				S1 = (Contour(p)-Ma)/(Mi-Ma) * (Pos(:,Tri(MiN,q))-Pos(:,Tri(MaN,q))) + Pos(:,Tri(MaN,q));
				S2 = (Mid-Contour(p))/(Mid-Ma) * (Pos(:,Tri(MaN,q))-Pos(:,Tri(MidN,q))) + Pos(:,Tri(MidN,q));
			else
				S1 = (Contour(p)-Ma)/(Mi-Ma) * (Pos(:,Tri(MiN,q))-Pos(:,Tri(MaN,q))) + Pos(:,Tri(MaN,q));
				S2 = (Contour(p)-Mid)/(Mi-Mid) * (Pos(:,Tri(MiN,q))-Pos(:,Tri(MidN,q))) + Pos(:,Tri(MidN,q));
			end
		H = [S1 S2]; % above the graph
		plot3(H(1,:),H(2,:),H(3,:),'Color',[0 0 0],'linewidth',3.5);
		end
	end
end
return

function RM = RotMatrix(Origin,Normal)
% Origin determines a point on the plane of intersection
% and Normal is the normal on the plane of intersection

% Assume algorithm works in z direction
% rotate system so it will be in the z-axis

if size(Origin,2) > 1, Origin = Origin'; end
if size(Normal,2) > 1, Normal = Normal'; end

% Translation matrix

% First determine the four dimensinal translation matrix
TrM = eye(4);
TrM([1:3],4) = -Origin;

% Rotation matrix

Normal = Normal/norm(Normal);
CosTheta0 = Normal(3);
SinTheta0 = sqrt(1-CosTheta0^2);
Rxy0 = norm(Normal([1 2]));
if Rxy0 > 0,
   CosPhi0 = Normal(1)/Rxy0;
   SinPhi0 = Normal(2)/Rxy0;
else
   CosPhi0 = 1;
   SinPhi0 = 0;
end
% rotate around z-axis with angle -Phi0
RPhi0 = [CosPhi0 SinPhi0 0;  -SinPhi0 CosPhi0 0; 0 0 1];
% rotate around y-axis with angle -Theta0
RTheta0 = [CosTheta0 0 (-SinTheta0); 0 1 0; SinTheta0 0 CosTheta0];
RotM = eye(4);
RotM([1:3],[1:3]) = RTheta0*RPhi0;

% first translation and then rotation
RM = RotM*TrM;
return

function Handle = PSGenTriLim(Pos,Tri,U,Origin,Normal)

% RM is a four dimensional rotation translation matrix

Handle = [];

hold on;

Pos4 = [Pos ; ones(1,size(Pos,2))];
RM = RotMatrix(Origin,Normal);
RMinv = inv(RM);

% Translate and rotate coordinates
Pos4 = RM*Pos4;

S = 1:size(Tri,2);
for k = 1:size(S,2),
	X = Pos4(1,Tri(:,S(k)));
	Y = Pos4(2,Tri(:,S(k)));
   Z = Pos4(3,Tri(:,S(k)));
   O = Pos4(4,Tri(:,S(k)));
   V = U(Tri(:,S(k)));
      
   F = find((Z < 0));
   if isempty(F),
      P = RMinv*[X; Y; Z; O];
      Handle(end+1) = patch(P(1,:),P(2,:),P(3,:),V','EdgeColor',[0.4 0.4 0.4]);
   end
         
   if (~isempty(F)&(length(Z)>length(F))),
      I = ones(1,length(Z));
      I(F) = 0;
      M = [ [I(2:end) I(1)] ; I ; I(end) I(1:end-1)];
      M = [1 2 4]*M;
      J = find(M);
      if ~isempty(J),
         Z = Z(J); Y = Y(J); X = X(J); V = V(J); M = M(:,J);
      end
            
      J = find(M == 1);
      if ~isempty(J),
	      I = mod(J,length(Z))+1;
         l = (Z(J))./(Z(J)-Z(I));
         Z(J) = 0; Y(J) = Y(J) + l*(Y(I)-Y(J)); X(J) = X(J) + l*(X(I)-X(J)); V(J) = V(J) +l*(V(I)-V(J));
      end
            
      J = find(M == 4);
      if ~isempty(J),
			I = mod(J-2,length(Z))+1;
         l = (Z(J))./(Z(J)-Z(I));
         Z(J) = 0; Y(J) = Y(J) + l*(Y(I)-Y(J)); X(J) = X(J) + l*(X(I)-X(J)); V(J) = V(J) + l*(V(I)-V(J));
      end
            
      J = find(M == 5);
      if ~isempty(J),
		   I1 = mod(J,length(Z))+1;   
        	I2 = mod(J-2,length(Z))+1;
            
       	l1 = (Z(J))./(Z(J)-Z(I1));
       	Z1 = zeros(1,length(J)); Y1 = Y(J) + l1*(Y(I1)-Y(J)); X1 = X(J) + l1*(X(I1)-X(J)); V1 = V(J) +l1*(V(I1)-V(J));
                 
         l2 = (Z(J))./(Z(J)-Z(I2));
         Z(J) = 0; Y(J) = Y(J) + l2*(Y(I2)-Y(J)); X(J) = X(J) + l2*(X(I2)-X(J)); V(J) = V(J) + l2*(V(I2)-V(J));
         Z2 = Z; Y2 = Y; X2 = X; V2 = V;
         L = length(Z);

         Z = []; Y = []; X = []; V = [];
         for r = 1:L,
            q = find(J == r);
            if ~isempty(q),
               Z = [Z Z2(r) Z1(q)]; Y = [Y Y2(r) Y1(q)]; X = [X X2(r) X1(q)]; V = [V V2(r) V1(q)];
            else
               Z = [Z Z2(r)]; Y = [Y Y2(r)]; X = [X X2(r)]; V = [V V2(r)];
            end
         end
      end
      O = ones(1,length(X));
      P = RMinv*[X; Y; Z; O];
      Handle(end+1) = patch(P(1,:),P(2,:),P(3,:),V','EdgeColor',[0.4 0.4 0.4]);
   end
end

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
