
%
%  The contents of this file are subject to the University of Utah Public
%  License (the "License"); you may not use this file except in compliance
%  with the License.
%
%  Software distributed under the License is distributed on an "AS IS"
%  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
%  License for the specific language governing rights and limitations under
%  the License.
%
%  The Original Source Code is SCIRun, released March 12, 2001.
%
%  The Original Source Code was developed by the University of Utah.
%  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
%  University of Utah. All Rights Reserved.
%
close all
clear all

hp='127.0.0.1:5505';

% transport(5,4,hp);

a=[0.1 0 1.5; 0 -0.2 0; 0 4.7 0.3];
% a=sparse(a);
transport(5,2,hp,a);
b=transport(2,1,hp);

% transport(5,5);
a
b
