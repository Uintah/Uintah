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


