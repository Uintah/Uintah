//static char *id="@(#) $Id$";

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

//
// $Log$
// Revision 1.8.2.3  2000/10/26 17:18:36  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.9  2000/06/06 16:01:44  dahart
// - Added get_triangles() to several classes for serializing triangles to
// send them over a network connection.  This is a short term (hack)
// solution meant for now to allow network transport of the geometry that
// Yarden's modules produce.  Yarden has promised to work on a more
// general solution to network serialization of SCIRun geometry objects. ;)
//
// Revision 1.8  1999/10/07 02:07:42  sparker
// use standard iostreams and complex type
//
// Revision 1.7  1999/09/04 06:01:48  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.6  1999/08/29 00:46:55  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/28 17:54:40  sparker
// Integrated new Thread library
//
// Revision 1.4  1999/08/19 23:18:05  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/17 23:50:21  sparker
// Removed all traces of the old Raytracer and X11 renderers.
// Also removed a .o and .d file
//
// Revision 1.2  1999/08/17 06:39:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:40  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:10:50  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:21  dav
// Import sources
//
//
