
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the input data from text files:                           %
%           1. time history of stresse                           %
%           2. time history of volumetric part of plastic strain %
%           3. time history of volumetric part of elastic strain %
%           4. time history of kappa                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           1. time history of stresse                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin1=fopen('outStress.txt','r');
t=[]; stress=[]; PlasStrain=[]; ElasStrain=[]; kappa=[];
word=fscanf(fin1,'%s1');
while word
    t=[t str2num(word)];
    fscanf(fin1,'%s1'); fscanf(fin1,'%s1'); fscanf(fin1,'%s1');
    stress=[stress [str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'))
        str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'))
        str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'))]];
    word=fscanf(fin1,'%s1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. time history of volumetric part of plastic strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin2=fopen('outPlasticStrain.txt','r');
word=fscanf(fin2,'%s1');
while word
    fscanf(fin2,'%s1'); fscanf(fin2,'%s1'); fscanf(fin2,'%s1');
    PlasStrain=[PlasStrain str2num(fscanf(fin2,'%s1'))];
    word=fscanf(fin2,'%s1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           3. time history of volumetric part of elastic strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin3=fopen('outElasticStrain.txt','r');
word=fscanf(fin3,'%s1');
while word
    fscanf(fin3,'%s1'); fscanf(fin3,'%s1'); fscanf(fin3,'%s1');
    ElasStrain=[ElasStrain str2num(fscanf(fin3,'%s1'))];
    word=fscanf(fin3,'%s1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           4. time history of kappa                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin4=fopen('outKappa.txt','r');
word=fscanf(fin4,'%s1');
while word
    fscanf(fin4,'%s1'); fscanf(fin4,'%s1'); fscanf(fin4,'%s1');
    kappa=[kappa str2num(fscanf(fin4,'%s1'))];
    word=fscanf(fin4,'%s1');
end
fclose('all');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw the analytical data:                 %
%           1. q-p plot (stress trajectory) %
%           2. crush curve                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           1. q-p plot (stress trajectory) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1); hold on;
plot(linspace(-204.12,204.12,100),61.238-0.3*linspace(-204.12,204.12,100),'-g','LineWidth',5);
pause
plot(linspace(-204.12,204.12,100),-61.238+0.3*linspace(-204.12,204.12,100),'-g','LineWidth',5);
pause
pValues=linspace(-612.35,-204.12,100);
Beta=sqrt(1-((3*pValues+612.35)/1224.8).^2);
plot(pValues,Beta*61.238-0.3*Beta.*pValues,'-g','LineWidth',5);
plot(pValues,-Beta*61.238+0.3*Beta.*pValues,'-g','LineWidth',5);

pause
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. crush curve                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
plot(linspace(-1837.0724,4*-1837.0724,100), ...
    0.5*exp(6.666666666666666e-4*(linspace(-1837.0724,4*-1837.0724,100)+1837.0724)),'-g','LineWidth',5);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw numerical results:                                %
%           1. q-p plot (stress trajectory)              %
%           2. crush curve                               %
%           3. time history of kappa                     %
%           4. pressure versus volumetric part of strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           1. q-p plot (stress trajectory)              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1); ylabel('q'); xlabel('p'); hold on; grid on;
p=(stress(1,:)+stress(5,:)+stress(9,:))/3;
sDev1=stress(1,:)-p; sDev2=stress(2,:);   sDev3=stress(3,:);
sDev4=stress(4,:);   sDev5=stress(5,:)-p; sDev6=stress(6,:);
sDev7=stress(7,:);   sDev8=stress(8,:);   sDev9=stress(9,:)-p;
J2=1/2*(sDev1.^2+sDev5.^2+sDev9.^2) ...
    +sDev2.^2+sDev6.^2+sDev3.^2;
J3=sDev1.*sDev5.*sDev9+2*sDev2.*sDev6.*sDev3 ...
    -sDev2.^2.*sDev9-sDev6.^2.*sDev5-sDev3.^2.*sDev1;
q=sqrt(3*J2).*sign(J3);
plot(p,q,'-b','LineWidth',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. crush curve                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2); ylabel('Porosity'); xlabel('I1'); hold on; grid on;
plot(3*p,PlasStrain+0.5,'-r','LineWidth',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           3. time history of kappa                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3); ylabel('Kappa'); xlabel('Time'); hold on; grid on;
plot(t,kappa,'-b','LineWidth',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           4. pressure versus volumetric part of strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4); ylabel('pressure'); xlabel('VolumetricStrain'); hold on; grid on;
plot(-ElasStrain-PlasStrain,-3*p,'-b','LineWidth',2);
