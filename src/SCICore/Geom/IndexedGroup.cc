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

#include <Geom/IndexedGroup.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

namespace SCICore {
namespace GeomSpace {

static Persistent* make_GeomIndexedGroup()
{
    return new GeomIndexedGroup;
}

PersistentTypeID GeomIndexedGroup::type_id("GeomIndexedGroup", "GeomObj",
					   make_GeomIndexedGroup);

GeomIndexedGroup::GeomIndexedGroup(const GeomIndexedGroup& /* g */)
{
    NOT_FINISHED("GeomIndexedGroup::GeomIndexedGroup");
}

GeomIndexedGroup::GeomIndexedGroup()
{
    // do nothing for now
}

GeomIndexedGroup::~GeomIndexedGroup()
{
    delAll();  // just nuke everything for now...
}

GeomObj* GeomIndexedGroup::clone()
{
    return scinew GeomIndexedGroup(*this);
}

void GeomIndexedGroup::reset_bbox()
{
    NOT_FINISHED("GeomIndexedGroup::reset_bbox");
}


void GeomIndexedGroup::get_bounds(BBox& bbox)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->get_bounds(bbox);
    }
}

void GeomIndexedGroup::get_bounds(BSphere& bsphere)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->get_bounds(bsphere);
    }
}

void GeomIndexedGroup::make_prims(Array1<GeomObj*>& free,
				  Array1<GeomObj*>& dontfree)
{
    HashTableIter<int, GeomObj*> iter(&objs);   
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->make_prims(free,dontfree);
    }	
}	

void GeomIndexedGroup::preprocess()
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->preprocess();
    }
}

void GeomIndexedGroup::intersect(const Ray& ray, Material* m,
				 Hit& hit)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->intersect(ray, m, hit);
    }
}

#define GEOMINDEXEDGROUP_VERSION 1

void GeomIndexedGroup::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("GeomIndexedGroup", GEOMINDEXEDGROUP_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, objs);
    stream.end_class();
}

bool GeomIndexedGroup::saveobj(ostream& out, const clString& format,
			       GeomSave* saveinfo)
{
  cerr << "saveobj IndexedGroup\n";
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	if(!obj->saveobj(out, format, saveinfo))
	    return false;
    }
    return true;
}

void GeomIndexedGroup::addObj(GeomObj* obj, int id)
{
    objs.insert(id,obj);
}

GeomObj* GeomIndexedGroup::getObj(int id)
{
    GeomObj *obj;
    if (objs.lookup(id,obj)) {
	return obj;
    }
    else {
	cerr << "couldn't find object in GeomIndexedGroup::getObj!\n";
    }
    return 0;
}

void GeomIndexedGroup::delObj(int id, int del)
{
    GeomObj* obj;

    if (objs.lookup(id,obj)) {
	objs.remove(id);
	//cerr << "Deleting, del=" << del << endl;
	if(del)
	    delete obj;
    }
    else {
	cerr << "invalid id in GeomIndexedGroup::delObj()!\n";
    }
}

void GeomIndexedGroup::delAll(void)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	delete obj;
    }

    objs.remove_all();
}

HashTableIter<int,GeomObj*> GeomIndexedGroup::getIter(void)
{
    HashTableIter<int,GeomObj*> iter(&objs);
    return iter;
}

HashTable<int,GeomObj*>* GeomIndexedGroup::getHash(void)
{
    return &objs;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
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

