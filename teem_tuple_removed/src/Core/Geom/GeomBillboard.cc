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
 * Billboard.cc: Pts objects
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Geom/GeomBillboard.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomBillboard()
{
    return scinew GeomBillboard(0, Point(0,0,0));
}

PersistentTypeID GeomBillboard::type_id("GeomBillboard", "GeomObj",
					make_GeomBillboard);


GeomBillboard::GeomBillboard(GeomHandle obj, const Point &p)
: GeomContainer(obj), at(p)
{

}

GeomObj* GeomBillboard::clone()
{
  cerr << "GeomBillboard::clone not implemented!\n";
  return 0;
}

void GeomBillboard::get_bounds(BBox& box)
{
  BBox bbox;

  child_->get_bounds(bbox);

  box.reset();
  box.extend( Point( bbox.min().x(), bbox.min().z(), bbox.min().y() ));
  box.extend( Point( bbox.max().x(), bbox.max().z(), bbox.max().y() ));
  box.translate( at.vector() );
}

#define GEOMBBOXCACHE_VERSION 1

void GeomBillboard::io(Piostream& stream)
{

    stream.begin_class("GeomBillboard", GEOMBBOXCACHE_VERSION);
    Pio(stream, child_);
    stream.end_class();
}

} // End namespace SCIRun


