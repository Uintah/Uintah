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
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>
#ifdef _WIN32
#include <float.h>
#define MAXDOUBLE DBL_MAX
#elif defined(__APPLE__)
   // no include
#else
#include <values.h>
#endif

namespace SCIRun {

using std::cerr;
using std::ostream;

static Persistent* make_GeomGroup()
{
    return scinew GeomGroup;
}

PersistentTypeID GeomGroup::type_id("GeomGroup", "GeomObj", make_GeomGroup);

GeomGroup::GeomGroup()
  : GeomObj()
{
}

GeomGroup::GeomGroup(const GeomGroup& copy)
  : GeomObj(copy), objs(copy.objs)
{
}

void GeomGroup::get_triangles( Array1<float> &v)
{
    for(unsigned int i=0;i<objs.size();i++)
	objs[i]->get_triangles(v);
}

void GeomGroup::add(GeomHandle obj)
{
    objs.push_back(obj);
}

void GeomGroup::remove(GeomHandle obj)
{
  vector<GeomHandle>::iterator oitr = objs.begin();
  while(oitr != objs.end())
  {
    if ((*oitr).get_rep() == obj.get_rep())
    {
      objs.erase(oitr);
      break;
    }
    ++oitr;
  }
}

void GeomGroup::remove_all()
{
   objs.clear();
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
    for(unsigned int i=0;i<objs.size();i++)
	objs[i]->get_bounds(in_bb);
}

void GeomGroup::reset_bbox()
{
    for(unsigned int i=0;i<objs.size();i++)
	objs[i]->reset_bbox();
}

#define GEOMGROUP_VERSION 2

void GeomGroup::io(Piostream& stream)
{

    const int version = stream.begin_class("GeomGroup", GEOMGROUP_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    if (version == 1 && stream.reading())
    {
      int del_children;
      Pio(stream, del_children);
    }
    Pio(stream, objs);
    stream.end_class();
}

} // End namespace SCIRun
