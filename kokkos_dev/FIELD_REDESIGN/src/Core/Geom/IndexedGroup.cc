//static char *id="@(#) $Id$";

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

#include <SCICore/Geom/IndexedGroup.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

#include <SCICore/Persistent/PersistentMap.h>

#include <iostream>
using std::cerr;
using std::ostream;

namespace SCICore {
namespace GeomSpace {

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
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomIndexedGroup", GEOMINDEXEDGROUP_VERSION);
				// Do the base class first...
    GeomObj::io(stream);
    Pio(stream, objs);
    stream.end_class();
}

//----------------------------------------------------------------------
bool GeomIndexedGroup::saveobj(ostream& out, const clString& format,
			       GeomSave* saveinfo)
{
    cerr << "saveobj IndexedGroup\n";
    MapIntGeomObj::iterator iter;
    for (iter = objs.begin(); iter != objs.end(); iter++) {
      if (!(*iter).second->saveobj(out, format, saveinfo)) return false;
    }
    return true;
}

//----------------------------------------------------------------------
void GeomIndexedGroup::addObj(GeomObj* obj, int id)
{
    objs[id] = obj;
}

//----------------------------------------------------------------------
GeomObj* GeomIndexedGroup::getObj(int id)
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
void GeomIndexedGroup::delObj(int id, int del)
{
  MapIntGeomObj::iterator iter = objs.find(id);
  if (iter != objs.end()) {
    //cerr << "Deleting, del=" << del << endl;
    if (del) delete (*iter).second;
    objs.erase(iter);
  }
  else {
    cerr << "invalid id in GeomIndexedGroup::delObj()!\n";
  }
}

//----------------------------------------------------------------------
void GeomIndexedGroup::delAll(void)
{
  MapIntGeomObj::iterator iter;
  for (iter = objs.begin(); iter != objs.end(); iter++) {
    delete (*iter).second;
  }
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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.7  2000/03/11 00:41:31  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.6  1999/10/07 02:07:48  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/29 00:46:57  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.4  1999/08/28 17:54:43  sparker
// Integrated new Thread library
//
// Revision 1.3  1999/08/17 23:50:31  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:19  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:49  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:55  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

