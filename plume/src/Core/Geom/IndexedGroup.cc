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
 *  IndexedGroup.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Geom/IndexedGroup.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Persistent/PersistentSTL.h>

#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

static Persistent* make_GeomIndexedGroup()
{
    return new GeomIndexedGroup;
}

PersistentTypeID GeomIndexedGroup::type_id("GeomIndexedGroup", "GeomObj",
					   make_GeomIndexedGroup);

//----------------------------------------------------------------------
GeomIndexedGroup::GeomIndexedGroup(const GeomIndexedGroup& /* g */)
{
    NOT_FINISHED("GeomIndexedGroup::GeomIndexedGroup");
}

//----------------------------------------------------------------------
GeomIndexedGroup::GeomIndexedGroup()
{
    // do nothing for now
}

//----------------------------------------------------------------------
GeomIndexedGroup::~GeomIndexedGroup()
{
    delAll();  // just nuke everything for now...
}

//----------------------------------------------------------------------
GeomObj* GeomIndexedGroup::clone()
{
    return scinew GeomIndexedGroup(*this);
}

//----------------------------------------------------------------------
void GeomIndexedGroup::reset_bbox()
{
    NOT_FINISHED("GeomIndexedGroup::reset_bbox");
}

//----------------------------------------------------------------------
void GeomIndexedGroup::get_bounds(BBox& bbox)
{
  MapIntGeomObj::iterator iter;
  for (iter = objs.begin(); iter != objs.end(); iter++) {
    (*iter).second->get_bounds(bbox);
  }  
}

#define GEOMINDEXEDGROUP_VERSION 1

//----------------------------------------------------------------------
void GeomIndexedGroup::io(Piostream& stream)
{

    stream.begin_class("GeomIndexedGroup", GEOMINDEXEDGROUP_VERSION);
				// Do the base class first...
    GeomObj::io(stream);
    Pio(stream, objs);
    stream.end_class();
}

//----------------------------------------------------------------------
void GeomIndexedGroup::addObj(GeomHandle obj, int id)
{
    objs[id] = obj;
}

//----------------------------------------------------------------------
GeomHandle GeomIndexedGroup::getObj(int id)
{
  MapIntGeomObj::iterator iter = objs.find(id);
  if (iter != objs.end()) {
    return (*iter).second;
  }
  else {
    cerr << "couldn't find object in GeomIndexedGroup::getObj!\n";
  }
  return 0;
}

//----------------------------------------------------------------------
void GeomIndexedGroup::delObj(int id)
{
  MapIntGeomObj::iterator iter = objs.find(id);
  if (iter != objs.end()) {
    objs.erase(iter);
  }
  else {
    cerr << "invalid id in GeomIndexedGroup::delObj()!\n";
  }
}

//----------------------------------------------------------------------
void GeomIndexedGroup::delAll()
{
  objs.clear();
}

//----------------------------------------------------------------------
GeomIndexedGroup::IterIntGeomObj GeomIndexedGroup::getIter(void)
{
  return IterIntGeomObj(objs.begin(), objs.end());
}

//----------------------------------------------------------------------
GeomIndexedGroup::MapIntGeomObj* GeomIndexedGroup::getHash(void)
{
  return &objs;
}

} // End namespace SCIRun


