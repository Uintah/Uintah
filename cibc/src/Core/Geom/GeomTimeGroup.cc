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
#include <sci_values.h>

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

