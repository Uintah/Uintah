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
 *  GeomBox.cc:  A box object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Geom/GeomBox.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomBox()
{
    return scinew GeomBox(Point(0,0,0), Point(1,1,1), 1);
}

PersistentTypeID GeomBox::type_id("GeomBox", "GeomObj", make_GeomBox);

GeomBox::GeomBox(const Point& p, const Point& q, int op) : GeomObj()
{

  min = Min( p, q );
  max = Max( p, q );

  for (int i=0; i<6; i++ )
    opacity[i] = op;
}

GeomBox::GeomBox(const GeomBox& copy)
: GeomObj(copy)
{
  min = copy.min;
  max = copy.max;
  for (int s=0; s<6; s++)
    opacity[s] = copy.opacity[s];
}

GeomBox::~GeomBox()
{
}

GeomObj* GeomBox::clone()
{
    return scinew GeomBox(*this);
}

void GeomBox::get_bounds(BBox& bb)
{
  bb.extend(min);
  bb.extend(max);
}

#define GEOMBOX_VERSION 1

void GeomBox::io(Piostream& stream)
{

    stream.begin_class("GeomBox", GEOMBOX_VERSION);
    GeomObj::io(stream);
    Pio(stream, min);
    Pio(stream, max);
    
    for ( int j=0; j<6; j++ )
      Pio(stream, opacity[j]);
    stream.end_class();
}

bool GeomBox::saveobj(ostream&, const string&, GeomSave*)
{
    NOT_FINISHED("GeomBox::saveobj");
    return false;
}

} // End namespace SCIRun


