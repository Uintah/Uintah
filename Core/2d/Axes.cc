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
 *  Axes.cc: 
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <stdio.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <sstream>
using std::ostringstream;

#include <Core/Geom/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Axes.h>
#include <Core/2d/Diagram.h>

namespace SCIRun {

Persistent* make_Axes()
{
  return scinew Axes;
}

PersistentTypeID Axes::type_id("Axes", "AxesObj", make_Axes);

Axes::Axes( Diagram *p, const string &name)
  : TclObj( "Axes" ), AxesObj(name), parent_(p), activepoly_(0),
    initialized_(false)
{
}


Axes::~Axes()
{
}


void
Axes::select( double x, double y, int b )
{
  cerr << "Axes select\n";
  AxesObj::select( x, y, b );
  //  update();
}
  
void
Axes::move( double x, double y, int b)
{
  AxesObj::move( x, y, b );
  //  update();
}
  
void
Axes::release( double x, double y, int b)
{
  AxesObj::release( x, y, b );
  //  update();
}
  

void
Axes::update()
{
}


#define AXES_VERSION 1

void 
Axes::io(Piostream& stream)
{
  stream.begin_class("Axes", AXES_VERSION);
  Widget::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  
