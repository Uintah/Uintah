/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2003, 2001, 1994 
  University of Utah. All Rights Reserved.
*/

//  The Macintosh does not run static object constructors when loading a 
//  dynamic library until a symbol from that library is referenced.  See:
//
//  http://wwwlapp.in2p3.fr/~buskulic/static_constructors_in_Darwin.html
//
//  We use this function to force Core/Datatypes to load and run constructors
//  (but only for Macs.)

void macForceLoad();
