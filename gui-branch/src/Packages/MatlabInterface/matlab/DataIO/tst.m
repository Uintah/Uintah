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

close all
clear all

hp='127.0.0.1:5517';
disp('test the engine');
transport(5,4,hp); % open client 

a=sparse([ 1 0 3 0 5]);
transport(5,2,hp,   'a=transport(2,1,hp);'   );
transport(5,2,hp,a);
transport(5,2,hp,   'a=a*2');
transport(5,2,hp,   'transport(2,2,hp,a);'   );
b=transport(5,1,hp);

b

transport(5,2,hp,   'exit');
transport(5,5); % close engine

