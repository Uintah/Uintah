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
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  DrawObj.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/2d/DrawObj.h>

namespace SCIRun {

PersistentTypeID DrawObj::type_id("DrawObj", "Datatype", 0);

DrawObj::DrawObj( const string &name) 
  : name_(name), parent_(0), ogl_(0)
{
}


string
DrawObj::tcl_color()
{
  char buffer[10];
  sprintf( buffer, "#%02x%02x%02x", 
	   int(color_.r()*255), int(color_.g()*255), int(color_.b()*255));
  buffer[7] = '\0';
  return string(buffer);
}

DrawObj::~DrawObj()
{
}

void DrawObj::reset_bbox()
{
  // Nothing to do, by default.
}

void DrawObj::io(Piostream&)
{
  // Nothing for now...
}

void Pio( Piostream & stream, DrawObj *& obj )
{
  Persistent* tmp=obj;
  stream.io(tmp, DrawObj::type_id);
  if(stream.reading())
    obj=(DrawObj*)tmp;
}

} // End namespace SCIRun
