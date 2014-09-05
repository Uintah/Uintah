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
 *  Drawable.cc: Displayable 2D object
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
#include <Core/2d/Drawable.h>

namespace SCIRun {

PersistentTypeID Drawable::type_id("Drawable", "Datatype", 0);

Drawable::Drawable( const string &name) 
  : name_(name), enabled_(true), parent_(0), ogl_(0)
{
  lock_ = scinew Mutex( name.c_str() );
}


Drawable::~Drawable()
{
}

void Drawable::reset_bbox()
{
  // Nothing to do, by default.
}

void Drawable::io(Piostream&)
{
  // Nothing for now...
}

void Pio( Piostream & stream, Drawable *& obj )
{
  Persistent* tmp=obj;
  stream.io(tmp, Drawable::type_id);
  if(stream.reading())
    obj=(Drawable*)tmp;
}

} // End namespace SCIRun
