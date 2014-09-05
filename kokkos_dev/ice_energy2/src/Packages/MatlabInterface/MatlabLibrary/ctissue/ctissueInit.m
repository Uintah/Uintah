function ctissueInit(rootdir)

% FUNCTION ctissueInit(rootdir)
%
% DESCRIPTION
% This function initializes the system for generating
% meshes of cardiac tissue
%
% INPUT
% rootdir    directory where meshes and models are generated
%
% OUTPUT
% -
%
% SEE ALSO -
 
%   For more information, please see: http://software.sci.utah.edu
%
%   The MIT License
%
%   Copyright (c) 2005 Scientific Computing and Imaging Institute,
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
 
    global CTISSUE;

    if ~isempty(CTISSUE), return; end

    where = mfilename('fullpath');
    where = where(1:(end-14));

    if nargin == 0 
      CTISSUE.root = where;
    else
      CTISSUE.root = rootdir;
    end
    
    if ~exists(rootdir,'dir')
      error(['directory ' rootdir 'does not exist']);
    end
    
    CTISSUE.temp = fullfile(CTISSUE.root,'temp');
    if ~exists(CTISSUE.temp,'dir'),
      mkdir(CTISSUE.root,'temp');
    end
    
    CTISSUE.models = fullfile(CTISSUE.root,'models');
    if ~exists(CTISSUE.models,'dir'),
      mkdir(CTISSUE.root,'models');
    end

    CTISSUE.parameters = fullfile(CTISSUE.root,'parameters');
    if ~exists(CTISSUE.parameters,'dir'),
      mkdir(CTISSUE.root,'parameters');
    end

    CTISSUE.cwmodel = fullfile(CTISSUE.root,'cwmodel');
    if ~exists(CTISSUE.cwmodel,'dir'),
      mkdir(CTISSUE.root,'cwmodel');
    end
    
return
