function m0=fcs(wordy,f,dd,sprt,mfit)

wm=wght(f,3);
[U,q,in]=nsing(f,wm,dd(:,1),mfit);
f=U(:,1:in)'*f;
ud=U(:,1:in)'*dd(:,1);

[m0,mp]=fcsinv(wordy-2,f,wm,ud,mfit,sprt,0);

%----------------------------------------------------------
%
% Focusing inversion for linear problem
% of three vector components
%
function [mm,mp]=fcsinv(wordy,F,wm,d,phi0,sprt,cgflag)

Nm=size(F,2);

if(~exist('sprt'))   sprt=0.5; end % This determines the value of support
if(~exist('cgflag')) cgflag=1; end

if(sprt>1)     error('fcsinv: sprt>1'); end
if(sprt<=1/Nm) error('fcsinv: sprt<=1/Nm'); end

bt=1e-8; % square root of machine precision

if(phi0<1e-8) error('fcsinv: Noise level is less than 1e-8'); end

m=ones(Nm,1);
w=m;

nd2=d'*d;

for it=1:20

 if(cgflag)  % solve iteratively

  m=m*0;
  r=-d;
  for k=1:10
   m=m+cnjgrdl(F,wm.*w,r,phi0/mf);
   r=F*m-d;
   mf=(r'*r)/nd2;
   if(wordy>1) disp(mf); end;
   if(mf<phi0) break; end;
  end

 else        % solve via SVD

  [U,Q2,U1]=svd(F*spdiags((wm.*w).^2,0,Nm,Nm)*F');
  q=sqrt(diag(Q2));
  du=(d'*U)';
  m=U*(du./(q.^2+udq2a(wordy-2,du,q,phi0*0.1)));
  m=((wm.*w).^2).*(m'*F)';

  % not soft enough
  % [U,q,in]=nsing(F,wm.*w,d,phi0*0.1);
  % m=U(:,1:in)*((U(:,1:in)'*d)./q(1:in).^2);
  % m=((wm.*w).^2).*(m'*F)';

 end

 mw3=reshape(sqrt(sum(reshape(m,Nm/3,3).^2,2))*[1 1 1],Nm,1);

 mx=max(abs(mw3));
 w1=(mw3.^2+(bt*mx)^2)*Nm;
 w2=mx*sum(abs(mw3));
 w=abs(mw3)*sqrt(Nm)./sqrt( (1-sprt + w1*sprt/w2) );

 s1=sum(mw3.^2./w1)*(1-sprt);
 s2=sum(mw3.^2)*sprt/w2;

 if(wordy>0)
  r=F*m-d;
  mf=(r'*r)/nd2;
  disp(['Misfit: ' num2str(mf)  ' Support: ' num2str(s1) ' L2 norm ' num2str(s2) ]);
 end

 if(nargout==2) % save intermediate results if requred
   mp(:,it)=m;  
   if(s1<sprt) mm=m; end;  
 else           % if not, quit when s1<sprt
   mm=m;
   if(s1<sprt) break; end;  
 end 

end

return

%----------------------------------------------------------
%
% Regularized conjugate gradient 
%
%
function mm=cnjgrdl(f,w,u,noise)

sa=sqrt( alpha0(noise^2,f,w,u) );

Nm=size(w,1);
Nd=size(u,1);

m=zeros(Nm+Nd,1);

s2old=1.;
rold=1e35;
h=zeros(Nm+Nd,1);

dn=u'*u;

for it=1:Nd

 conjug=u'*u;
 n=m(Nm+[1:Nd]);
 r=u-sa*n;

 estn=sa*sa*n'*n;

 disp([ 'time it sqer estn=' num2str([toc it sqrt([conjug estn]/dn) ] ) ]); 

 if(conjug<estn) break; end;
 if(sqrt(r'*r/dn)<noise) break; end;

 s=[ w.*(u'*f)' ; sa*u ];

 snorm=s'*s;

 if snorm ==0 ; break; end;
 if conjug >= rold; break; end;

 h=h+(snorm/s2old-1)*h;
 h=h+s;

 s2old=snorm;
 rold=conjug;

 fh=f*(w.*h(1:Nm)) +sa*h(Nm+[1:Nd]);
 fh2=fh'*fh;

 step=sum(fh.*u)/fh2;
 m=m-step*h;
 u=u-step*fh;

end

mm=w.*m(1:Nm);

return;

%----------------------------------------------------------
%
% Estimate of alpha for grossly underdetermined problem
%
%
function alp=alpha0(phi1,f,w,r)

 s=w.*(r'*f)';
 p=f*(w.*s);

 a=(r'*r)*(1-phi1);
 b=(s'*s)*(1-2*phi1);
 c=(p'*p)*(0-phi1);
 alp= (-b+sqrt(b*b-4*a*c) )/(2*a) ;

return;


%
% Compute number of singular values necessary 
% to deliver given level of misfit
%
function [U,q,in]=nsing(f,wm,d,mfit)

[Nd,Nm]=size(f);

[U,Q2,U1]=svd(f*spdiags(wm.^2,0,Nm,Nm)*f');

du=U'*d;
q=sqrt(diag(Q2));
nd2=(du'*du);

for k=1:Nd
 mv(1:Nd,1)=0;
 mv(1:k)=du(1:k)./q(1:k);
 r=q.*mv-du;
 mf(k)=(r'*r)/nd2;
end

[tmp,in]=min(abs(mf-mfit));
in=length((find(q(1:in)~=0)));

return;
% UDQ2A  Find alpha for underdetermined problem
%        problem assumed to be SVDable
% 
% by Oleg Portniaguine, oleg@cs.utah.edu
% 2001Oct02
%
% syntax: 
%
%   aq=udq2a(wordy,du,q,phi0)
%
% call example (below)
%
%%--------------------------------------
%%  Prepare F and data
%%--------------------------------------
% rand('seed',1234);
% Nm=100;
% Nd=50;
% F=rand(Nd,Nm);
% [U,Q,V]=svd(F);
% Q(30:end,30:end)=0;
% F=U*Q*V';
%
% m0=rand(Nm,1);
% d=F*m0;
% d=d+0.01*norm(d);
%%--------------------------------------
%
% [U,Q2,U1]=svd(F*F',0);
% q=sqrt(diag(Q2));
% m=F'*U*( (U'*d)./(q.^2+udq2a(1,U'*d,q,0.01)) );
% mfit=sqrt( sum( (F*m-d).^2 )/(d'*d) );
%
%%-----------end of example-------------
%
% Theory. Consider:
% 
% 1a) ||Fm-d||^2+aq*||m||^2=min
% 1b) m=F'*inv(F*F'+aq*I)*d
%
% use SVD. Remember, it is fast in Matlab-6
%
% 2a) F=U*Q*V'
% 2b) U*Q^2*U'=F*F' 
% 2c) U'*U=I
% 2d) V'*V=I
% 2e) q=diag(Q) 
%
% Insert (2) into (1b)
%
% 3a) m=F'*inv(U*Q^2*U'+aq*I)*d
% 3b) inv(A*B*C)=inv(C)*inv(B)*inv(A)
% 3c) m=F'*U*inv(Q^2+aq*I)*U'*d
%
% To estimate alpha, diagonalize
% insert (2) into (1a)
%
% 4a) ||U*Q*V'*m-d||^2+aq*||m||^2=min
% 4b) mv=V'*m
% 4c) mv'*mv=m'*m
% 4d) ||U*Q*V'*m-d||=||Q*V'*m-U'*d||
% 4e) du=U'*d
% 4f) ||Q*mv-du||^2+aq*||mv||^2=min
%
% Normalization, reshuffle (4f)
%
% 6a) aq=a*max(q)
% 6b) ( || diag(q)*mv - du ||^2 + a*max(q)^2*||mv||^2  )/||du||^2 = min
% 6c)  qm=q/max(q)
% 6d)  mq=mv*max(q)/||du||
% 6e)  dn=du/||du||
% 
% Inserting (6c,d,e) into (6b) produces the formula
% this code uses to estimate a
%
% 6f)  || diag(qm)*mq - dn ||^2 + a*||mq||^2 = min
% 
% Method: solve two auxilary problems
%
% 7a)  || diag(qm)*mq1 - dn - diag(qm)*dn ||^2 + a*||mq1||^2 = min
% 7b)  || diag(qm)*mq2 - dn + diag(qm)*dn ||^2 + a*||mq1||^2 = min
% 7c)  mq=(qm1+qm2)/2
% 7d)  r1=diag(qm)*mq1 - dn;
% 7e)  r2=diag(qm)*mq2 - dn;
% 7f)  r3=(r1+r2)/2;
% 7g)  r12=(||r1||-||r2||)^2;
%
% if a problem has exact solution, (7g) serves as estimate of misfit.
% If not, (7g) decreases anyway
% thus, alpha is taken where 7g is less than phi0
%
%
function aq=udq2a(wordy,du,q,phi0)

dn=du/sqrt(du'*du);
qm=q/max(q);

%----------------------------------------------
for a=1:30
 ea=exp(-a);
 mq1=diag((qm-qm.^2)./(qm.^2+ea))*dn;
 mq2=diag((qm+qm.^2)./(qm.^2+ea))*dn;
 r1=(diag(qm)*mq1-dn);
 r2=(diag(qm)*mq2-dn);
 r3=(r1+r2)/2;
 nr1(a)=r1'*r1;
 nr2(a)=r2'*r2;
 nr3(a)=r3'*r3;
 n12(a)=abs(sqrt(nr2(a))-sqrt(nr1(a))).^2;
end

% return value 

[tmp,in]=min(abs(n12-phi0));
aq=exp(-in)*max(q)^2;

% debug prints

if(wordy>0) % debug prints
 semilogy(nr1,'r');
 hold on;
 semilogy(nr2,'g');
 semilogy(nr3,'b');
 semilogy(n12,'m');

%mq=qm./(qm.^2+aq/max(q)^2).*dn;
%mv=q./(q.^2+aq).*du;
%[mq mv*max(q)/sqrt(du'*du)]

end

return;


%
% Compute sensitivity weight
%
%
function wm=wght(f,flag)

% compute norms

Nm=size(f,2);
wm=zeros(Nm,1);
for im=1:Nm
 wm(im)=sum(f(:,im).^2);
end

switch(flag)
 case 1, % scalar weights

     wm=1./sqrt(sqrt(wm));

 break;
 case 3, % weights for 3D vectors

     wm=reshape(wm,Nm/3,3);
     w3=sum(wm,2);
 
     for k=1:3
       wm(:,k)=sqrt(sqrt(wm(:,k))./w3);
     end
     wm=wm(:);

 break;
 otherwise
   error('unknown weighting type');
end
return;

