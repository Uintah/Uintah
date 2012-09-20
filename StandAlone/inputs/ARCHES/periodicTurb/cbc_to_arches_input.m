% function: cbc_to_arches_input
% author:   Tony Saad
% date:     September, 2012
%
% cbc_to_arches_input: reads the cbc_uvw data and generates data that is
% proper for Arches to initialize.
% The cbc_uvw dataset represents an
% initial condition for isotropic turbulence that fits the cbc 
% (Comte-Bellot & Corrsin) experimental spectrum. This data was generated
% using a two-step process:
% 1. Generate random phases and fit them to the cbc spectrum
% 2. Since the random data does not satisfy mass conservation, it is run
% through a single timestep subject to Navier-Stokes physics
% 3. Compute the energy spectrum of the velocity field after a single
% timestep and inject some energy into those modes that showed energy
% decay. Note that at this stage the velocity will gain some divergence but
% it is of the order 10^-3.
%
% For more information on this procedure, see Rnady McDermott's dissertation.
%
% The cbc_uvw data may be found on the FDS code website:
% http://code.google.com/p/fds-smv/source/browse/trunk/FDS/trunk#trunk%2FVe
% rification%2FTurbulence
%
% Acknowledgment: Randy McDermott's feedback has been essential in
% understanding how this dataset is formatted.
%
function cbc_to_arches_input(uvwFileName)

M = csvread(uvwFileName); % read the csv data into a matrix
s = size(M);
n = round(s(1)^(1/3));    % get the dimension

% convert to 3D array - note the addition of the plus face
u = zeros(n+1,n,n);
v = zeros(n,n+1,n);
w = zeros(n,n,n+1);

p=0;
for k=1:n
    for j=1:n
        for i=2:n+1
            p=p+1;
            u(i,j,k) = M(p,1);
        end
    end
end

p=0;
for k=1:n
    for j=2:n+1
        for i=1:n
            p=p+1;
            v(i,j,k) = M(p,2);
        end
    end
end

p=0;
for k=2:n+1
    for j=1:n
        for i=1:n
            p=p+1;
            w(i,j,k) = M(p,3);
        end
    end
end

% account for periodicity - this was the key to decoding this dataset
u(1,:,:) = u(n+1,:,:);
v(:,1,:) = v(:,n+1,:);
w(:,:,1) = w(:,:,n+1);

fileName = strcat(uvwFileName,'_arches.txt');
fileID = fopen(fileName,'w');
fprintf( fileID,'%i %i %i\n',n, n, n );
for k=1:n
    for j=1:n
        for i=1:n
            fprintf( fileID,'%.16f %.16f %.16f %.16f\n',u(i,j,k),v(i,j,k),w(i,j,k) );
        end
    end
end
