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

#include <Core/Geom/GeomTimeGroup.h>
#include <Core/Containers/Array2.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>
#ifdef _WIN32
#include <float.h>
#define MAXDOUBLE DBL_MAX
#elif !defined __APPLE__
#include <values.h>
#endif

namespace SCIRun {

using std::cerr;
using std::ostream;

static Persistent* make_GeomTimeGroup()
{
    return scinew GeomTimeGroup;
}

PersistentTypeID GeomTimeGroup::type_id("GeomTimeGroup", "GeomObj", make_GeomTimeGroup);

GeomTimeGroup::GeomTimeGroup()
  : GeomObj()
{
}

GeomTimeGroup::GeomTimeGroup(const GeomTimeGroup& copy)
  : GeomObj(copy), objs(copy.objs), start_times(copy.start_times)
{
}

void GeomTimeGroup::add(GeomHandle obj,double time)
{
    objs.push_back(obj);
    start_times.push_back(time);
}

void GeomTimeGroup::remove(GeomHandle obj)
{
  vector<GeomHandle>::iterator oitr = objs.begin();
  vector<double>::iterator titr = start_times.begin();
  while(oitr != objs.end())
  {
    if ((*oitr).get_rep() == obj.get_rep())
    {
      objs.erase(oitr);
      start_times.erase(titr);
      break;
    }
    ++oitr;
    ++titr;
  }
}

void GeomTimeGroup::remove_all()
{
    objs.clear();
    start_times.clear();
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

    for(unsigned int i=0;i<objs.size();i++)
	objs[i]->get_bounds(in_bb);

#if 0
    in_bb.extend(bbox);
#endif
}


void GeomTimeGroup::setbbox(BBox& b)
{
  bbox = b;
}

void GeomTimeGroup::reset_bbox()
{
    for(unsigned int i=0;i<objs.size();i++)
	objs[i]->reset_bbox();
}

#define GEOMTimeGroup_VERSION 2

void GeomTimeGroup::io(Piostream& stream)
{

    const int ver = stream.begin_class("GeomTimeGroup", GEOMTimeGroup_VERSION);
    // Do the base class first...
    GeomObj::io(stream);
    if (ver == 1 && stream.reading())
    {
      int del_children;
      Pio(stream, del_children);
    }
    Pio(stream, objs);
    Pio(stream,start_times);
    stream.end_class();
}

} // End namespace SCIRun

