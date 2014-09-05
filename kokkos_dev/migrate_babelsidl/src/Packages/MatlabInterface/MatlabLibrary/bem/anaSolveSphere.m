function [U] = anaSolveSphere(option,Pos,DipPos,DipStr,Radius,Sigma,n)

% FUNCTION U = anaSolveSphere(option,Pos,DipPos,DipStr,Radius,Sigma,n)
% 
% DESCRIPTION
% This general function evaluates almost "everyting" for a multilayer sphere
%
% The potential is evaluate using an expansion in n Legendre polynomials, the current 
% density mapping uses a derivative of this expansion in polynomials. The magnetic field is
% computed using an analytical expression.
%
% For the Magnetic field Radius and Sigma need to be supplied although they do not influence the 
% outcome, n does not need to be specified either.
% The Magnetic field is computed using the Magnetic potential outside the source regions
% The solution is therefore only valid outside the volume
%
% The Current density will only give a valid result inside the modelling space and the potential
% will produce valid values everywhere, both inside and outside the volume conductor.
%
% The Potential and the current density are evaluated using an expansion in Legendre polynomials. 
% The more, the more accurate the result will be.
%
% Choose option 'B' for magnetic field, 'U' for potential and 'J' for current density
%
% INPUT
% option     'U', 'B' or 'J'
% Pos         Positions [3xM] at which U, J or B needs to be evaluated
% DipPos      Position of Dipole needs to be in inner sphere (except for magnetic field)
% DipStr      Dipole strength (a vector with three components)
% Radius      Radii of the spheres from inside to outside
% Sigma       Conductivities in same order
% n           Number of Legendre polynomials to evaluate for potential (optional)
%             default value for n = 40;
%             All vectors are rotate if needed, so do not bother about the format.
% 
% OUTPUT      
% U           The potential U, the current density J or the magnetic field B
%

% JG STINSTRA 2002

% first a couple of checks to assure correct input

if size(Pos,1) == 1, Pos = Pos'; end

if size(DipPos,1) > 1, DipPos = DipPos'; end
if size(DipStr,1) > 1, DipStr = DipStr'; end
if size(Radius,1) > 1, Radius = Radius'; end
if size(Sigma,1) > 1, Sigma = Sigma'; end

if length(Radius) < 1, error('no sphere defined'); end
if length(Radius) == length(Sigma), Sigma = [Sigma 0]; end
if (length(Radius)+1) ~= length(Sigma), error('Radius and Sigma need to be of the same size'); end

if nargin == 6, n = 40; end

if norm(DipPos) > Radius(1), error('Dipole not in inner sphere'); end

% define relative accuracy
disp('relative accuary set to 1e-8');
e = 1e-8;

% Start by determining the angle of the dipole

R0 = norm(DipPos);

if R0 > 0
   
	CosTheta0 = DipPos(3)/R0;
	SinTheta0 = sqrt(1-CosTheta0^2);
	R0xy = norm(DipPos([1 2]));

	if R0xy > e*R0,
		CosPhi0 = DipPos(1)/R0xy;
	   SinPhi0 = DipPos(2)/R0xy;
	else
   	% system symmetric around z-axis so I'll have to define a phi angle
	   % does not matter what I choose
   	CosPhi0 = 1;
	   SinPhi0 = 0;
	end

	% Rotate system to align dipole axis with z-axis
	% rotate around z-axis with angle -Phi0
	RPhi0 = [CosPhi0 SinPhi0 0;  -SinPhi0 CosPhi0 0; 0 0 1];

	% rotate around y-axis with angle -Theta0
	RTheta0 = [CosTheta0 0 (-SinTheta0); 0 1 0; SinTheta0 0 CosTheta0];
	Rot = RTheta0*RPhi0;

else
   
   % Rotate dipole strength now to z-axis is easier in solution
   
   CosTheta0 = DipStr(3)/norm(DipStr);
	SinTheta0 = sqrt(1-CosTheta0^2);
	R0xy = norm(DipStr([1 2]));

	if R0xy > e*R0,
		CosPhi0 = DipStr(1)/R0xy;
	   SinPhi0 = DipStr(2)/R0xy;
	else
   	% system symmetric around z-axis so I'll have to define a phi angle
	   % does not matter what I choose
   	CosPhi0 = 1;
	   SinPhi0 = 0;
	end

	% Rotate system to align dipole axis with z-axis
	% rotate around z-axis with angle -Phi0
   RPhi0 = [CosPhi0 SinPhi0 0;  -SinPhi0 CosPhi0 0; 0 0 1];
   
	% rotate around y-axis with angle -Theta0
	RTheta0 = [CosTheta0 0 (-SinTheta0); 0 1 0; SinTheta0 0 CosTheta0];
   Rot = RTheta0*RPhi0;
end

% Rotate complete system

Pos = Rot*Pos;
DipPos = Rot*DipPos';
DipStr = Rot*DipStr';

InvRot = inv(Rot);

% System now completely rotated
% now determine goniometric functions for the positions in Pos

R = sqrt(sum(Pos.*Pos));
CosTheta = Pos(3,:)./R;
SinTheta = sqrt(1-CosTheta.^2);
Rxy = sqrt(sum(Pos([1 2],:).*Pos([1 2],:)));

Index = find((Rxy-e*R) > 0);
if length(Index) > 0,
    CosPhi(Index) = Pos(1,Index)./Rxy(Index);
    SinPhi(Index) = Pos(2,Index)./Rxy(Index);
end
nIndex = find((Rxy-e*R)<= 0);
if length(nIndex) > 0,
    CosPhi(nIndex) = 1;
    SinPhi(nIndex) = 0;
end

% The whole system is not rotated and aligned along the z-axis

switch option
   
case 'B'
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate the magnetic field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Magnetic evaluate 
% This is programmed accoording to formulas given by Stok

if R0 == 0,
   U = zeros(3,size(R,2));
else
   
	mu04pi = 1e-7; % mu0 divided by 4 pi
	RR0 = sqrt(R.^2 - 2*R0*R.*CosTheta + R0^2);

	Index = find(SinTheta > e);
	A(Index) = (1./(R0*SinTheta(Index).^2)).*(((R0*CosTheta(Index)-R(Index))./RR0(Index))+1);

	nIndex = find(SinTheta <= e); % DO exceptions (sintheta approaches zero)
	A(nIndex) = R0./(2*(R(nIndex) - (sign(CosTheta(nIndex))*R0)).^2);

	Br = mu04pi*(R0*SinTheta)./(RR0.^3).*(DipStr(1)*SinPhi - DipStr(2)*CosPhi);
	Btheta = mu04pi*(DipStr(1)*SinPhi-DipStr(2)*CosPhi).*(R0*(R0-R.*CosTheta)./(RR0.^3) + CosTheta.*A);
	Bphi = -mu04pi*(DipStr(1)*CosPhi+DipStr(2)*SinPhi).*A; % minus SinTheta as we divide later by it so avoiding problem by dividing by zero

	% Back to carthesian coordinates
	Bx = Br.*SinTheta.*CosPhi + Btheta.*CosTheta.*CosPhi./R - Bphi.*SinPhi./R;
	By = Br.*SinTheta.*SinPhi + Btheta.*CosTheta.*SinPhi./R + Bphi.*CosPhi./R;
	Bz = Br.*CosTheta - Btheta.*SinTheta./R;

	% add everything into one matrix

	B = [Bx;By;Bz];

	Index = find(R <= Radius(end));
	B(:,Index) = NaN*ones(3,length(Index));

	U = InvRot*B;

   % end magnetic field calculation
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate the potential distribution

case {'U','J'}

U = [];

% reverse both R and Sigma 
% due to old algorithm where this was reversed
Sigma = Sigma(end:-1:1);
Radius = Radius(end:-1:1);

% Algoritme voor het potentiaal in een sphere met meerdere lagen
% Dit algoritme is gebaseerd op de formules van J.C. de Munck
% Potentiaal benaderen door n Legendere polynomen

N = size(Sigma,2);
cA = zeros(N,2,n);
cB = zeros(N,2,n);

% n is geheel binnen in
% bereken A(2) en B(2)'s

% evaluate the coefficients
for p = 1:n;
	cA(1,2,p) = 0;
	cB(1,2,p) = 1;
	for j = 1:(N-1)
		sr = Sigma(j)/Sigma(j+1);
		M = [ (p+1+p*sr)/(2*p+1)  (p+1)*(1-sr)/(2*p+1) ; p*(1-sr)/(2*p+1) (p+sr*(p+1))/(2*p+1)];
		C(1,1) = cA(j,2,p);
		C(2,1) = cB(j,2,p)/(Radius(j)^(2*p+1));
		D = M*C;
		cA(j+1,2,p) = D(1,1);
		cB(j+1,2,p) = D(2,1)*Radius(j)^(2*p+1);
	end
	cA(N,1,p) = 1;
	cB(N,1,p) = 0;
end

% Calculate U

switch option,
   
case 'U'
   
% Fill in coefficients and calculate the potential   

U = zeros(1,size(Pos,2));

if R0 == 0,
   n = 1; %only one expansion is enough
end

for p = 1:n, % for each polynomial
	R2 = zeros(1,size(Pos,2));
	for k = 1:N-2 % for each volume
		T = find((R <= Radius(k))&(R > Radius(k+1)));
		R2(T) = cA(k+1,2,p)*R(T).^p + cB(k+1,2,p)*R(T).^(-p-1);
	end
	T = find((R > Radius(1)));
	R2(T) = cA(1,2,p)*R(T).^p+cB(1,2,p)*R(T).^(-p-1);
	T = find((R <= Radius(N-1)));
	R2(T) = cA(N,2,p)*R(T).^p+cB(N,2,p)*R(T).^(-p-1);

	if R0 == 0,
    	R1d = cA(N,1,p);
   else
      R1 = cA(N,1,p)*R0^p+cB(N,1,p)*R0^(-p-1);
		R1d = cA(N,1,p)*R0^(p-1)*p - cB(N,1,p)*R0^(-p-2)*(p+1);
   end
   
      
	P = legendre(p,CosTheta);
	Yn00 =  P(1,:);
   Yn10x = P(2,:).*CosPhi;
   Yn10y = P(2,:).*SinPhi;
   if R0 == 0,
      U = U + R2/(4*pi*Sigma(N)*cB(N,2,p)).*(DipStr(3)*R1d*Yn00);
   else
      U = U + R2/(4*pi*Sigma(N)*cB(N,2,p)).*(DipStr(3)*R1d*Yn00-DipStr(1)*R1/R0*Yn10x-DipStr(2)*R1/R0*Yn10y);
   end
end

case 'J'
   
J = zeros(3,size(Pos,2));

if R0 == 0,
   n = 1; %only one expansion is enough
end

for p = 1:n,
   R2 = zeros(1,size(Pos,2));
   dR2 = R2;
   Sigmas = zeros(1,size(Pos,2));
   for k = 1:N-2
		T = find((R <= Radius(k))&(R > Radius(k+1)));
		R2(T) = cA(k+1,2,p)*R(T).^p + cB(k+1,2,p)*R(T).^(-p-1);
      dR2(T) = (p*cA(k+1,2,p)*R(T).^(p-1) - (p+1)*cB(k+1,2,p)*R(T).^(-p-2));   
      Sigmas(T) = Sigma(k+1);
   end
	T = find((R > Radius(1)));
   R2(T) = cA(1,2,p)*R(T).^p+cB(1,2,p)*R(T).^(-p-1);
   dR2(T) = (p*cA(1,2,p)*R(T).^(p-1) - (p+1)*cB(1,2,p)*R(T).^(-p-2));
   T = find((R <= Radius(N-1)));
	R2(T) = cA(N,2,p)*R(T).^p+cB(N,2,p)*R(T).^(-p-1);
   dR2(T) = (p*cA(N,2,p)*R(T).^(p-1) - (p+1)*cB(N,2,p)*R(T).^(-p-2));
   Sigmas(T) = Sigma(N);
   
   GradR2 = [dR2.*SinTheta.*CosPhi ; dR2.*SinTheta.*CosTheta ; dR2.*CosTheta];
   R2 = [R2; R2; R2];
   Sigmas = [Sigmas ; Sigmas ; Sigmas];
   
	R1 = cA(N,1,p)*R0^p+cB(N,1,p)*R0^(-p-1);
	R1d = cA(N,1,p)*R0^(p-1)*p - cB(N,1,p)*R0^(-p-2)*(p+1);
   
	P = legendre(p,CosTheta);
	Yn00 =  [1 1 1]'*P(1,:);
   Yn10x = [1 1 1]'*(P(2,:).*CosPhi);
   Yn10y = [1 1 1]'*(P(2,:).*SinPhi);
   
   % In case n=1 for matlab n =2 as one is add to the order of polynomial
   % Then no m = 2 is present to avoid problems the next if statement is used
   
   P2 = P(2,:);
   if p >2,
      P3 = P(3,:);
   else
      P3 =zeros(size(P2));
   end
      
   GradYn00 = [CosTheta.*CosPhi.*P2./R; CosTheta.*SinPhi.*P2./R; -SinTheta.*P2./R];
   
   %Do Grad Yn10x
   
   % Deal with a couple of exceptions in P(2,:)/SinPhi
   % For SinPhi this results in a divide by zero when it is zero
   % However Pn(2,:) = -sin(Phi) d/dx P0n(x)
   % Using this and the fact that P0n(1) = 1 and P0n(-1) = (-1)^n
   % and as well x d/dx[Pn(x)] - d/dx[Pn-1(x)] = nPn(x)
   % These equations will predict the values  for the exceptions
   % The corrected results are in PSin
   
   T = find(SinTheta ~= 0);
   pT = find(SinTheta == 0 & CosTheta > 0);
   nT = find(SinTheta == 0 & CosTheta < 0);
   
   if ~isempty(T), PSin(T) = P2(T)./SinTheta(T); end
   if ~isempty(pT), PSin(pT) = -sum(1:p); end   
   if ~isempty(nT), PSin(nT) = sum(1:p)*(-1)^p; end   

   GradYn10x= [(CosTheta.^2.*CosPhi.^2+SinPhi.^2).*PSin./R + (CosTheta.*CosPhi.^2).*P3./R;
      -(SinPhi.*SinTheta.*CosPhi.*P2)./R + (CosTheta.*SinPhi.*CosPhi.*P3./R);
      CosTheta.*CosPhi.*P2./R + SinTheta.*CosPhi.*P3./R];
   
   
   GradYn10y= [ -(SinPhi.*SinTheta.*CosPhi.*P2)./R + (CosTheta.*SinPhi.*CosPhi.*P3./R);
      (CosTheta.^2.*SinPhi.^2+CosPhi.^2).*PSin./R + (CosTheta.*SinPhi.^2).*P3./R;
      CosTheta.*SinPhi.*P2./R + SinTheta.*SinPhi.*P3./R];
   
   % calculate derivatives of these
   if R0 ==0,
      J = J - GradR2/(4*pi*Sigma(N)*cB(N,2,p)).*(DipStr(3)*R1d*Yn00) - R2/(4*pi*Sigma(N)*cB(N,2,p)).*(DipStr(3)*R1d*GradYn00);
   else
      J = J - GradR2/(4*pi*Sigma(N)*cB(N,2,p)).*(DipStr(3)*R1d*Yn00-DipStr(1)*R1/R0*Yn10x-DipStr(2)*R1/R0*Yn10y) - R2/(4*pi*Sigma(N)*cB(N,2,p)).*(DipStr(3)*R1d*GradYn00-DipStr(1)*R1/R0*GradYn10x-DipStr(2)*R1/R0*GradYn10y);
   end
   
end

J = Sigmas.*J;
U = InvRot*J;   
   
end
end

U = U';
return
