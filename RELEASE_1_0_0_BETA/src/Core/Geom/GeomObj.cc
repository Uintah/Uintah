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
 *  Geom.cc: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {

PersistentTypeID GeomObj::type_id("GeomObj", "Persistent", 0);

GeomObj::GeomObj(int id) : id(id),
  _id(0x1234567,0x1234567,0x1234567)
{
}

GeomObj::GeomObj(IntVector i)
  :id( 0x1234567 ), _id(i)
{
}

GeomObj::GeomObj(int id_int, IntVector i)
  :id( id_int ), _id(i)
{
}

GeomObj::GeomObj(const GeomObj&)
{
}

GeomObj::~GeomObj()
{
}

void GeomObj::get_triangles( Array1<float> &)
{
  cerr << "GeomObj::get_triangles - no triangles" << endl;
}

void GeomObj::reset_bbox()
{
    // Nothing to do, by default.
}

void GeomObj::io(Piostream&)
{
    // Nothing for now...
}

void Pio( Piostream & stream, GeomObj *& obj )
{
    Persistent* tmp=obj;
    stream.io(tmp, GeomObj::type_id);
    if(stream.reading())
	obj=(GeomObj*)tmp;
}

} // End namespace SCIRun
