% Title: Table Probe
% Author: Jeremy Thornock
% Date: Feb. 10, 2009
% Descriptioon: This script will create a surface or contour plot of the 
% mixing and reaction table for a given variable.  
% How to run this: Just navigate to where your table is and run it. 
% The user specifies the required information at run time.  

close all
clear all
clc

tableName = input('Enter the table name: ','s');
fid = fopen(tableName);
checkComments = true;
%Go through the comments at the top of the table
while (checkComments == true)
    line = fgetl(fid);
    if (size(findstr(line,'#'),1) == 0)
        checkComments = false;
    end
end
   
%Now I assume the next five lines are consistent among all tables
% we can through away this stuff (or maybe make a smarter script?)
for i = 1:4
    line = fgetl(fid);
end

%get the number of points
line = fgetl(fid); 
numPoints = textscan(line,'%n'); temp = numPoints{1};
Nmf = temp(1); Nhl = temp(2); Nsv = temp(3); 

%get the number of species
line = fgetl(fid);
Nsp = str2num(line);

%get the species list
line = fgetl(fid);
speciesList = textscan(line,'%s');

%get the units list
line = fgetl(fid);

%next line is empty
line = fgetl(fid); %should be empty

%get the heat loss range
line = fgetl(fid);
hl = textscan(line,'%n');hl = hl{1};

%get scalar variance range
line = fgetl(fid);
sv = textscan(line,'%n');sv = sv{1};

%this line is blank
line = fgetl(fid); %should be empty

fprintf('Species in table:\n');
speciesList{1}

fprintf('Enter a species that you want plotted ...\n')
whichsp = input('vs. Mixture fraction: ','s');
fprintf('a) Fixed Heat Loss \n');
fprintf('b) Fixed Scalar Variance \n');
whichplot = input('Enter a choice ( a or b ): ','s');
whichhl = 0;
whichsv = 0;
if ( whichplot=='a')
    whichhl = input('For which heat loss: ');
elseif ( whichplot == 'b')
    whichsv = input('For which scalar variance: ');
end

%find the location of the species
ii = 0;
spIndex = 0;
for i = 1:Nsp
    ii = ii + 1;
    if (strcmp(whichsp,speciesList{1}{i}))
        spIndex = ii;
    end
end

if (spIndex == 0) 
    fprintf('ERROR! Could not find species: %15s \n', whichsp)
    fprintf(' so exiting...\n')
    stop
else
    fprintf('You requested: %15s \n', whichsp)
    fprintf('and I found: %15s \n', speciesList{1}{spIndex});
    fprintf('continuing!\n')
end

%find the location of the heat loss value
hlIndex = find(hl >= whichhl);
if (hlIndex(1) > 1 && hlIndex(1) < Nhl)
    hlIndex = hlIndex(1) -1;
else
    hlIndex = hlIndex(1);
end

%find the location of the scalar variance
svIndex = find(sv >= whichsv);
if (svIndex(1) > 1 && svIndex(1) < Nsv)
    svIndex = svIndex(1) -1;
else
    svIndex = svIndex(1);
end

%Here is where we parse the rest of the table.
for i = 1:Nsp
    mf = textscan(fid,'%f',Nmf);

    recordme = false;
    if (i==spIndex)
        recordme = true;
    end

    ii = 0;
    for j = 1:Nhl
        %read the entire heat loss block
        hlblock = textscan(fid,'%f',Nmf*Nsv);

        if (j == hlIndex && recordme == true && whichplot == 'a')
            temp = hlblock{1};
            temp = reshape(temp,Nmf,Nsv);
            temp = temp';
            plotme = temp;
            break
        elseif (recordme == true && whichplot == 'b')
            ii = ii + 1;
            temp = hlblock{1};
            temp = reshape(temp,Nmf,Nsv);
            temp = temp';
            plotme(ii,:) = temp(svIndex,:);
        end
    end
    
    if (recordme == true)
        break
    end

end

fclose(fid); %close the table

%now put the values in something easy to plot
% bb = 1;
% be = Nmf;
% temp = plotme{:};
% for i = 1:Nsv
%     A(i,:) = temp(bb:be);
%     bb = be+1;
%     be = be+Nmf;
% end

mixfrac = mf{1};

fprintf('Enter a choice of the following\n');
fprintf('a) Filled contour plot\n');
fprintf('b) Surface plot\n');
plotChoice = input('Choice: ','s');

if (whichplot == 'a')
    if (plotChoice == 'a')
        contourf(mixfrac,sv',plotme,16);
    elseif (plotChoice == 'b')
        surf(mixfrac,sv',plotme);
    end
    set(gca,'FontSize',16);
    xlabel('Mixture Fraction')
    ylabel('Scalar Variance')
    zlabel(whichsp)
elseif (whichplot == 'b')
    if (plotChoice == 'a')
        contourf(mixfrac,hl',plotme,16);
    elseif (plotChoice == 'b')
        surf(mixfrac,hl',plotme);
    end
    set(gca,'FontSize',16);
    xlabel('Mixture Fraction')
    ylabel('Heat Loss')
    zlabel(whichsp)
end
colorbar
fprintf('Your *block* of values are stored in the variable: plotme\n')










