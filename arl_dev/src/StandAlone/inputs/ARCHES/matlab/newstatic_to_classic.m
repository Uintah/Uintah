close all
clear all

tableName = input('Enter the table name: ','s');
fid = fopen(tableName);
wfid = fopen('new_table.mxn','w');
fprintf('This should produce a table called: new_table.mxn\n');
checkComments = true;
%Go through the comments at the top of the table
while (checkComments == true)
    line = fgetl(fid);
    if (size(findstr(line,'#'),1) == 0)
        checkComments = false;
    else 
        fprintf(wfid, '%s', line);
        fprintf(wfid, '\n','');
    end
end
   
%Now I assume the next five lines are consistent among all tables
% we can throw away this stuff (or maybe make a smarter script?)

%fuel and oxidizer enthalpy
line = fgetl(fid); 
fuel_ox_enthalpy = textscan(line, '%f');
foh = fuel_ox_enthalpy{:};

%the rest not used; 
for i = 1:3
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
unitList = textscan(line, '%s');

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

%Here is where we parse the rest of the table.
for i = 1:Nsp
    mf = textscan(fid,'%f',Nmf);

    for j = 1:Nhl
        
        %read the entire heat loss block
        T{i,j} = textscan(fid,'%f',Nmf*Nsv);

    end
   
end

mf_new = mf{:}; 
numP = numPoints{:};
t_numP = numP;
numP(2) = t_numP(3); 
numP(3) = t_numP(2); 

speciesList{1}{Nsp+1} = 'adiabaticenthalpy';
unitList{1}{Nsp+1} = '-';
Nsp = Nsp + 1;  

add_debug = false; 
if ( add_debug == true ) 
    speciesList{1}{Nsp+1} = 'debug_species';
    unitList{1}{Nsp+1} = '-';
    Nsp = Nsp + 1; 
end

%replace some names with the new naming convention: 
for i = 1:Nsp
   if (strcmp(speciesList{1}{i},'heat_capacity'))
       speciesList{1}{i} = 'specificheat';
   end
   if (strcmp(speciesList{1}{i},'sensible_heat'))
       speciesList{1}{i} = 'sensibleenthalpy'; 
   end
end

sp = speciesList{:}; 
unit = unitList{:};

fprintf(wfid, '%u \n', 3);
fprintf(wfid, '%s %s %s', 'mixture_fraction scalar_variance heat_loss');
fprintf(wfid, '\n','');
fprintf(wfid, '%u %u %u \n', numP);
fprintf(wfid, '%d \n', Nsp);
for i = 1:size(sp,1)
    pp = sp(i); 
    fprintf(wfid, '%20s ', char(pp));
end
fprintf(wfid, '\n','');
for i = 1:size(unit,1)
    uu = unit(i); 
    fprintf(wfid, '%15s ', char(uu));
end
fprintf(wfid, '\n','');
fprintf(wfid, '\n','');


%rewrite the table in the correct format   
fprintf(wfid, '%10.7e ',hl);
fprintf(wfid, '\n', '');
fprintf(wfid, '%10.7e ',sv);
fprintf(wfid, '\n', '');

%-- old format --
%fprintf(wfid, '\n','');
%fprintf(wfid, '%10.7e ', mf_new);
%fprintf(wfid, '\n', ' ');
%fprintf(wfid, '\n', ' ');

factor = 1; 
if ( add_debug ) 
    factor = 2; 
end

for i = 1:Nsp-factor
    
    %write mf points

    
    for j = 1:Nhl
        
        %-- new format -- 
        fprintf(wfid, '\n','');
        fprintf(wfid, '%10.7f ', mf_new);
        fprintf(wfid, '\n', ' ');
        fprintf(wfid, '\n', ' ');
        
        block = T{i,j};
        block = block{:}; 
        istart = 1;
        iend = Nmf; 
        %write block
        for k = 1:Nsv
            
            fprintf(wfid, '%10.5e  ', block(istart:iend));
            fprintf(wfid, '\n', '');
            istart = istart + Nmf; 
            iend = iend + Nmf; 
            
        end
        
        %-- old format -- 
        %fprintf(wfid, '\n', '' ); 
        
        
        
    end
    %-- old format -- 
    %fprintf(wfid, '%10.7e ', mf_new);
    %fprintf(wfid, '\n', ' ');
    %fprintf(wfid, '\n', ' ');
end

% add the adiabatic enthalpy:
for i=1:size(mf_new,1)
    adiab_enthalpy(i) = mf_new(i)*foh(1)+(1-mf_new(i))*foh(2);
end
%adiab_enthalpy = mf_new*foh(1)+(1.0-mf_new).*foh(1);
for j = 1:Nhl
    
    %-- new format --
    fprintf(wfid, '\n','');
    fprintf(wfid, '%10.7f ', mf_new);
    fprintf(wfid, '\n', ' ');
    fprintf(wfid, '\n', ' ');
    
    for k = 1:Nsv
        
        fprintf( wfid, '%10.5e ', adiab_enthalpy(:));
        fprintf( wfid, '\n','');
        
    end
end

if ( add_debug == true )
    
    A = ones(size(mf_new,1),1)*2; 
    B = ones(size(mf_new,1),1)*5;
    F = betapdf(mf_new,A,B);
    debug_species = ones(size(mf_new,1),1)*1.2 - F/2.5;
    
    for i = 1:size(mf_new,1)
       if (mf_new(i) > .2)
          factor = (mf_new(i)/0.8-.2)*.1; 
          debug_species(i) = debug_species(i)-factor; 
       end
    end
    
    for j = 1:Nhl
        
        %-- new format --
        fprintf(wfid, '\n','');
        fprintf(wfid, '%10.7f ', mf_new);
        fprintf(wfid, '\n', ' ');
        fprintf(wfid, '\n', ' ');
        
        hlfactor = ( 1.5 - hl(j) )/2.0; 
        
        for k = 1:Nsv
            
            svfactor = .1 + .1*sin(2*pi*sv(k));
            
            fprintf( wfid, '%10.5e ', debug_species(:)*hlfactor*svfactor);
            fprintf( wfid, '\n','');
            
        end
    end
end
fclose(wfid)