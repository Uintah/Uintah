/*
 *  Group.cc:  Groups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;
#ifdef _WIN32
#include <float.h>
#define MAXDOUBLE DBL_MAX
#else
#include <values.h>
#endif

namespace SCICore {
namespace GeomSpace {

using SCICore::Math::Min;
using SCICore::Containers::Array2;

static Persistent* make_GeomGroup()
{
    return scinew GeomGroup;
}

PersistentTypeID GeomGroup::type_id("GeomGroup", "GeomObj", make_GeomGroup);

GeomGroup::GeomGroup(int del_children)
: GeomObj(), objs(0, 100), del_children(del_children)
{
}

GeomGroup::GeomGroup(const GeomGroup& copy)
: GeomObj(copy), del_children(copy.del_children)
{
    objs.grow(copy.objs.size());
    for(int i=0;i<objs.size();i++){
	GeomObj* cobj=copy.objs[i];
	objs[i]=cobj->clone();
    }
}

void GeomGroup::get_triangles( Array1<float> &v)
{
    for(int i=0;i<objs.size();i++)
	objs[i]->get_triangles(v);
}

void GeomGroup::add(GeomObj* obj)
{
    objs.add(obj);
}

void GeomGroup::remove(GeomObj* obj)
{
   for(int i=0;i<objs.size();i++)
      if (objs[i] == obj) {
	 objs.remove(i);
	 if(del_children)delete obj;
	 break;
      }
}

void GeomGroup::remove_all()
{
   if(del_children)
      for(int i=0;i<objs.size();i++)
	 delete objs[i];
   objs.remove_all();
}

int GeomGroup::size()
{
    return objs.size();
}

GeomObj* GeomGroup::clone()
{
    return scinew GeomGroup(*this);
}

void GeomGroup::get_bounds(BBox& in_bb)
{
    for(int i=0;i<objs.size();i++)
	objs[i]->get_bounds(in_bb);
}

GeomGroup::~GeomGroup()
{
    if(del_children){
	for(int i=0;i<objs.size();i++)
	    delete objs[i];
    }
}

void GeomGroup::reset_bbox()
{
    for(int i=0;i<objs.size();i++)
	objs[i]->reset_bbox();
}

#define GEOMGROUP_VERSION 1

void GeomGroup::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;

    stream.begin_class("GeomGroup", GEOMGROUP_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, del_children);
    SCICore::Containers::Pio(stream, objs);
    stream.end_class();
}

bool GeomGroup::saveobj(ostream& out, const clString& format,
			GeomSave* saveinfo)
{
    static int cnt = 0;
    cnt++;
    cerr << "saveobj Group " << cnt << "\n";

    for(int i=0;i<objs.size();i++){
	if(!objs[i]->saveobj(out, format, saveinfo))
	  { cnt--;
	    return false;
	  }
    }
    cerr << "saveobj Group done " << cnt << "\n";
    cnt--;
    return true;
}

} // End namespace GeomSpace
} // End namespace SCICore
