//static char *id="@(#) $Id$";

/*
 *  TimeGroup.cc:  TimeGroups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomTimeGroup.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#ifdef _WIN32
#include <float.h>
#define MAXDOUBLE DBL_MAX
#else
#include <values.h>
#endif

namespace SCICore {
namespace GeomSpace {

static Persistent* make_GeomTimeGroup()
{
    return scinew GeomTimeGroup;
}

PersistentTypeID GeomTimeGroup::type_id("GeomTimeGroup", "GeomObj", make_GeomTimeGroup);

GeomTimeGroup::GeomTimeGroup(int del_children)
: GeomObj(), objs(0, 100), start_times(0,100),del_children(del_children)
{
}

GeomTimeGroup::GeomTimeGroup(const GeomTimeGroup& copy)
: GeomObj(copy), del_children(copy.del_children)
{
    objs.grow(copy.objs.size());
    start_times.grow(copy.start_times.size());
    for(int i=0;i<objs.size();i++){
	GeomObj* cobj=copy.objs[i];
	objs[i]=cobj->clone();
	objs[i]->set_parent(this);
	start_times[i] = copy.start_times[i];
    }
}

void GeomTimeGroup::add(GeomObj* obj,double time)
{
    obj->set_parent(this);
    objs.add(obj);
    start_times.add(time);
}

void GeomTimeGroup::remove(GeomObj* obj)
{
   for(int i=0;i<objs.size();i++)
      if (objs[i] == obj) {
	 objs.remove(i);
	 start_times.remove(i);
	 if(del_children)delete obj;
	 break;
      }
}

void GeomTimeGroup::remove_all()
{
   if(del_children)
      for(int i=0;i<objs.size();i++)
	 delete objs[i];
   objs.remove_all();
   start_times.remove_all();
}

int GeomTimeGroup::size()
{
    return objs.size();
}

GeomObj* GeomTimeGroup::clone()
{
    return scinew GeomTimeGroup(*this);
}

void GeomTimeGroup::get_bounds(BBox& in_bb)
{

    for(int i=0;i<objs.size();i++)
	objs[i]->get_bounds(in_bb);

#if 0
    in_bb.extend(bbox);
#endif
}

void GeomTimeGroup::get_bounds(BSphere& in_sphere)
{
  for(int i=0;i<objs.size();i++)
    objs[i]->get_bounds(in_sphere);
}


void GeomTimeGroup::make_prims(Array1<GeomObj*>& free,
			 Array1<GeomObj*>& dontfree)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->make_prims(free, dontfree);
    }
}

GeomTimeGroup::~GeomTimeGroup()
{
    if(del_children){
	for(int i=0;i<objs.size();i++)
	    delete objs[i];
    }
}

void GeomTimeGroup::setbbox(BBox& b)
{
  bbox = b;
}

void GeomTimeGroup::reset_bbox()
{
    for(int i=0;i<objs.size();i++)
	objs[i]->reset_bbox();
}

void GeomTimeGroup::preprocess()
{
    int i;
    for(i=0;i<objs.size();i++){
	objs[i]->preprocess();
    }
}

void GeomTimeGroup::intersect(const Ray& ray, Material* matl,
			  Hit& hit)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->intersect(ray, matl, hit);
    }
}

#define GEOMTimeGroup_VERSION 1

void GeomTimeGroup::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("GeomTimeGroup", GEOMTimeGroup_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, del_children);
    Pio(stream, objs);
    Pio(stream,start_times);
    stream.end_class();
}

bool GeomTimeGroup::saveobj(ostream& out, const clString& format,
			    GeomSave* saveinfo)
{
    static int cnt = 0;
    cnt++;
    cerr << "saveobj TimeGroup " << cnt << "\n";

    for(int i=0;i<objs.size();i++){ cerr << cnt << ">";
	if(!objs[i]->saveobj(out, format, saveinfo))
	  { cnt--;
	    return false;
	  }
    }
    cnt--;
    return true;
}

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:14  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:45  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:53  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//
