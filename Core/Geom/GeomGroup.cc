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
#include <sci_values.h>

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
