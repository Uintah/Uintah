/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
  long k = ((long)ptr) >> 2; // Disgard unused bits, word aligned pointers.
  return (int)((k^(3*GEOM_LOCK_POOL_SIZE+1))%GEOM_LOCK_POOL_SIZE);
}   

GeomObj::GeomObj() :
  ref_cnt(0),
  lock(*(lock_pool.getMutex(lock_pool_hash(this)))),
  id_int_(0x1234567),
  id_intvector_(0x1234567,0x1234567,0x1234567),
  id_longlong_((long long)(0x1234567))
{
}


GeomObj::GeomObj(const GeomObj&obj) :
  ref_cnt(0),
  lock(*(lock_pool.getMutex(lock_pool_hash(this)))),
  id_int_(obj.id_int_),
  id_intvector_(obj.id_intvector_),
  id_longlong_(obj.id_longlong_)
{
}

GeomObj::~GeomObj()
{
}


bool
GeomObj::getId( int& id )
{
  if ( id_int_ == 0x1234567)
    return false;
  else {
    id = id_int_;
    return true;
  }
}

bool
GeomObj::getId( IntVector& id )
{
  if ( id_intvector_ == IntVector(0x1234567,0x1234567,0x1234567) )
    return false;
  else {
    id = id_intvector_;
    return true;
  }
}

bool
GeomObj::getId( long long& id )
{
  if ( id_longlong_ == (long long)(0x1234567))
    return false;
  else {
    id = id_longlong_;
    return true;
  }
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
