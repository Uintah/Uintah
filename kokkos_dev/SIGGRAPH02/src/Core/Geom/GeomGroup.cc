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

#include <Core/Geom/GeomGroup.h>
#include <Core/Containers/Array2.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;
#ifdef _WIN32
#include <float.h>
#define MAXDOUBLE DBL_MAX
#else
#include <values.h>
#endif

namespace SCIRun {


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

    stream.begin_class("GeomGroup", GEOMGROUP_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    Pio(stream, del_children);
    Pio(stream, objs);
    stream.end_class();
}

bool GeomGroup::saveobj(ostream& out, const string& format,
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

} // End namespace SCIRun
