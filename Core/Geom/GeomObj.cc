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
#include <Core/Thread/MutexPool.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {

PersistentTypeID GeomObj::type_id("GeomObj", "Persistent", 0);

#define GEOM_LOCK_POOL_SIZE 257
MutexPool lock_pool("GeomObj pool", GEOM_LOCK_POOL_SIZE);

static int lock_pool_hash(GeomObj *ptr)
{
  long k = ((long)ptr) >> 3; // Disgard unused bits, byte aligned pointers.
  return (int)((k^(3*GEOM_LOCK_POOL_SIZE+1))%GEOM_LOCK_POOL_SIZE);
}   

GeomObj::GeomObj(int id) :
  ref_cnt(0),
  lock(*(lock_pool.getMutex(lock_pool_hash(this)))),
  id(id),
  _id(0x1234567,0x1234567,0x1234567)
{
}

GeomObj::GeomObj(IntVector i) :
  ref_cnt(0),
  lock(*(lock_pool.getMutex(lock_pool_hash(this)))),
  id( 0x1234567 ),
  _id(i)
{
}

GeomObj::GeomObj(int id_int, IntVector i) :
  ref_cnt(0),
  lock(*(lock_pool.getMutex(lock_pool_hash(this)))),
  id( id_int ),
  _id(i)
{
}

GeomObj::GeomObj(const GeomObj&) :
  ref_cnt(0),
  lock(*(lock_pool.getMutex(lock_pool_hash(this))))
  // TODO: id and _id uninitialized.
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
