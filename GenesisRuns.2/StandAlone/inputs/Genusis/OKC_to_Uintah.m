#! /usr/bin/env octaveWrap
%_________________________________
% This octave file generates the parallelpiped geometry object used by uintah.  
%
% Taking an input file with each box defined with the following points
%
%         /                           %
%       P5-----------------P6         %
%      / \                /  \        %
%    P1...\....(x).......P2   \       %
%      \   \              \    \      %
%      (z)  P8-----------------P7     %
%        \  /               \  /      %
%         \/      Front      \/       %
%         P4-----------------P3       %
%
% The parallelpiped geometry object is specified 
% using these points
%                                     %
%                                     % 
%       *------------------*          % 
%      / \                / \         % 
%    P3...\..............*   \        % 
%      \   \             .    \       % 
%      (z)  P2-----------------*      % 
%        \  /             .   /       % 
%         \/               . /        % 
%         P1-------(x)-----P4         % 
%______________________________________________________________________
%
%  Format of the input file is:
%   1             2              3            4             5              6            7             8              9       
%#  p1.x          p1.y           p1.z         p2.x          p2.y           p2.z         p3.x          p3.y           p3.z    <snip>
%0.782814	0.000826	-0.664714	0.800582	0.000826	-0.633938	0.823485	0.000826	-0.647161    <snip>
%0.585762	0.000826	-0.832464	0.597332	0.000826	-0.812424	0.613078	0.000826	-0.821515    <snip>
%__________________________________
%  usage  ./OKC_to_Uintah.m OKC_rotated_3d.dat
%%__________________________________

clear all;
close all;
format short e;
unix('/bin/rm OKC_clean OKC_mpm.xml OKC_ice.xml');



inputfile = argv();

%clean out any lines with the character (#)
cmd = sprintf('grep -v "#" %s > OKC_clean 2>&1', inputfile{1});
[s,r] = system( cmd );

points = load("OKC_clean");
nBuildings = length( points(:,24) );

%__________________________________
%  open file and print header
printf("______________________________________________________________________\n");
fid = fopen("OKC_mpm.xml",'w');
fprintf(fid, "<?xml version='1.0' encoding='ISO-8859-1' ?>\n");
fprintf(fid, "<Uintah_Include>\n");
fprintf(fid, "              <union>\n");

counter = 0;
for (b = 1:nBuildings)
  P1 = { points(b,1),  points(b,2),  points(b,3) };
  P2 = { points(b,4),  points(b,5),  points(b,6) };
  P3 = { points(b,7),  points(b,8),  points(b,9) };
  P4 = { points(b,10), points(b,11), points(b,12) };
  P5 = { points(b,13), points(b,14), points(b,15) };
  P6 = { points(b,16), points(b,17), points(b,18) };
  P7 = { points(b,19), points(b,20), points(b,21) };
  P8 = { points(b,22), points(b,23), points(b,24) };
  
  fprintf(fid,"                <parallelepiped label = \"%d\">\n",counter)
  fprintf(fid,"                    <p1>  [%f, %f, %f]  </p1>\n", P4{1}, P4{2}, P4{3}) 
  fprintf(fid,"                    <p2>  [%f, %f, %f]  </p2>\n", P8{1}, P8{2}, P8{3})
  fprintf(fid,"                    <p3>  [%f, %f, %f]  </p3>\n", P1{1}, P1{2}, P1{3})
  fprintf(fid,"                    <p4>  [%f, %f, %f]  </p4>\n", P3{1}, P3{2}, P3{3})
  fprintf(fid,"                </parallelepiped>\n") 
  counter +=1;
end
fprintf(fid, "              </union>\n");
fprintf(fid, "</Uintah_Include>");
fclose(fid);

unix("more  OKC_mpm.xml", "-echo");

%__________________________________
% now output ICE geometry object
printf("\n\n______________________________________________________________________\n");
fid = fopen("OKC_ice.xml", 'w');
fprintf(fid, "<?xml version='1.0' encoding='ISO-8859-1' ?>\n");
fprintf(fid, "<Uintah_Include>\n");
fprintf(fid, "              <union>\n");

counter = 0;
for (b = 1:nBuildings)
  fprintf(fid,"                <parallelepiped label = \"%d\">     </parallelepiped>\n",counter)
  counter +=1;
end

fprintf(fid, "              </union>\n");
fprintf(fid, "</Uintah_Include>");
fclose(fid);
unix("more  OKC_ice.xml","-echo");
unix('/bin/rm OKC_clean');
exit
