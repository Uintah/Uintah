function [P, Region, numRegions] = initialConditions()
globalParams


%______________________________________________________________________
if(0)
% Material properties (ideal gas)
P.maxTime           = 5e-4;                         % Maximum simulation time [sec]
P.cv                = 717.5;                        % Specific_heat
P.gamma             = 1.4;                          % gamma coefficient in the Equation of State (EOS)


%================ Partition of the domain into regions ================
numRegions          = 2;                            % Partition of the domain into numRegions regions; physical properties are uniform within each region
Region              = cell(numRegions,1);           % This array holds the geometrical partition physical info
count               = 0;

%********** Parameters of the high density left region **********
count               = count+1;
R.label             = 'leftpartition';              % Title of this region
R.min               = 0;                            % Location of lower-left corner of this region [length]
R.max               = 0.5;                          % Location of upper-right corner of this region [length]

R.velocity          = 0.0;                          % Initial velocity
R.temperature       = 300.0;                        % Initial temperature [Kelvin]
R.density           = 1.1768292682926831000;        % Initial density
R.pressure          = 101325.0;                     % Initial pressure (1 atmosphere)

Region{count}       = R;                            % Add region to list
%********** Parameters of the low density right region **********
count               = count+1;
R.min               = 0.5;                          % Location of lower-left corner of this region [length]
R.max               = 1;                            % Location of upper-right corner of this region [length]

R.velocity          = 0.0;                          % Initial velocity
R.temperature       = 300.0;                        % Initial temperature [Kelvin]
R.density           = 0.11768292682926831000;       % Initial density
R.pressure          = 10132.5;                      % Initial pressure (0.1 atmosphere)
Region{count}       = R;                            % Add region to list

end


%______________________________________________________________________
% non-dimensionalized shock tube problem: 
%  src/orderAccuracy/test_config_files/ICE/riemann.ups
%     <delt_max>           0.0005       </delt_max>
%     <delt_init>          1.0e-20      </delt_init>
%     <outputTimestepInterval> 1        </outputTimestepInterval>
%     <cfl>               0.25          </cfl>
%     second order advection

if(1)

% Material properties (ideal gas)
P.maxTime           = 0.2;                         % Maximum simulation time [sec]
P.cv                = 1.0;                          % Specific_heat
P.gamma             = 1.4;                          % gamma coefficient in the Equation of State (EOS)


%================ Partition of the domain into regions ================
numRegions          = 2;                            % Partition of the domain into numRegions regions; physical properties are uniform within each region
Region              = cell(numRegions,1);           % This array holds the geometrical partition physical info
count               = 0;

%********** Parameters of the high density left region **********
count               = count+1;
R.label             = 'leftpartition';              % Title of this region
R.min               = 0;                            % Location of lower-left corner of this region [length]
R.max               = 0.3;                          % Location of upper-right corner of this region [length]

R.velocity          = 0.75;                         % Initial velocity
R.temperature       = 2.5;                          % Initial temperature [Kelvin]
R.density           = 1.0;                          % Initial density
R.pressure          = 1.0;                          % Initial pressure (1 atmosphere)

Region{count}       = R;                            % Add region to list
%********** Parameters of the low density right region **********
count               = count+1;
R.min               = 0.3;                          % Location of lower-left corner of this region [length]
R.max               = 1;                            % Location of upper-right corner of this region [length]

R.velocity          = 0.0;                          % Initial velocity
R.temperature       = 2.0;                          % Initial temperature [Kelvin]
R.density           = 0.125;                        % Initial density
R.pressure          = 0.1;                          % Initial pressure (0.1 atmosphere)
Region{count}       = R;                            % Add region to list

end


end
