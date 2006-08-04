function scirunMountLibrary(varargin)

% FUNCTION scirunMountLibrary
%
% DESCRIPTION
% This function mounts the Library of Matlab functions
% distributed with SCIRun. The function adds all the 
% paths to the directories where these functions can be
% found. These functions are as well under subversion
% control and are automatically available when running
% the matlab-engine. In case one wants to use the library
% separately, just execute this function inside matlab
% and all functionality will be available.
%
% Apart from adding the proper paths this function will
% create a global called SCIRUN with all information on the
% function library contained in it. This one can be used
% to store custom data as well as to look up version
% numbers etc.
% 
% INPUT -
%
% OUTPUT -
%
% SEE ALSO -

%   For more information, please see: http://software.sci.utah.edu
%
%   The MIT License
%
%   Copyright (c) 2004 Scientific Computing and Imaging Institute,
%   University of Utah.
%
%   License for the specific language governing rights and limitations under
%   Permission is hereby granted, free of charge, to any person obtaining a
%   copy of this software and associated documentation files (the "Software"),
%   to deal in the Software without restriction, including without limitation
%   the rights to use, copy, modify, merge, publish, distribute, sublicense,
%   and/or sell copies of the Software, and to permit persons to whom the
%   Software is furnished to do so, subject to the following conditions:
%
%   The above copyright notice and this permission notice shall be included
%   in all copies or substantial portions of the Software.
%
%   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
%   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
%   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
%   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
%   DEALINGS IN THE SOFTWARE.

   global SCIRUN;

   % Did we already mount the library in this Matlab session?
   if isfield(SCIRUN,'version'), return; end

   % Where are we?
   % Try to find my location.

   fullprogramname = which(mfilename('fullpath')); 

   % The file runs thus the file must exist
   [programpath,programname] = fileparts(fullprogramname);

   % Put this info up there for every one to use
   SCIRUN.path = programpath;

   % Display a message of mounting the SCIRun-Matlab function
   % library.

   startupdoc = fullfile(SCIRUN.path,'scirun','startup.txt');

   startuptext = [];
   fid = fopen(startupdoc); 
   if fid ~= -1, 
      SCIRUN.version = fgets(fid);
      while ~feof(fid), startuptext = [startuptext fgets(fid)]; end
      fclose(fid);
      disp(startuptext);
   end
   
  % All directories in this function library are automatically added to the 
  % path.
	
   mydir = dir(programpath);

   for p = 1:length(mydir),
      if mydir(p).isdir == 1,
         % Since matlab does not add a path twice we do not need to worry about doing it twice
         addpath(fullfile(SCIRUN.path,mydir(p).name,'')); 
      end
   end

   return