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
 *  PointCloudMesh.cc: PointCloudField mesh
 *
 *  Written by:
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <float.h>  // for DBL_MAX
#include <iostream>


namespace SCIRun {

using namespace std;


PersistentTypeID PointCloudMesh::type_id("PointCloudMesh", "Mesh", maker);

BBox
PointCloudMesh::get_bounding_box() const
{
  BBox result;

  Node::iterator i, ie;
  begin(i);
  end(ie);
  for ( ; i != ie; ++i)
  {
    result.extend(points_[*i]);
  }

  return result;
}


void
PointCloudMesh::transform(const Transform &t)
{
  vector<Point>::iterator itr = points_.begin();
  vector<Point>::iterator eitr = points_.end();
  while (itr != eitr)
  {
    *itr = t.project(*itr);
    ++itr;
  }
}


bool
PointCloudMesh::locate(Node::index_type &idx, const Point &p) const
{
  Node::iterator ni, nie;
  begin(ni);
  end(nie);

  idx = *ni;

  if (ni == nie)
  {
    return false;
  }

  double closest = (p-points_[*ni]).length2();

  ++ni;
  for (; ni != nie; ++ni)
  {
    if ( (p-points_[*ni]).length2() < closest )
    {
      closest = (p-points_[*ni]).length2();
      idx = *ni;
    }
  }

  return true;
}


int
PointCloudMesh::get_weights(const Point &p, Node::array_type &l, double *w)
{
  Node::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


PointCloudMesh::Node::index_type
PointCloudMesh::add_point(const Point &p)
{
  points_.push_back(p);
  return points_.size() - 1;
}

#define PointCloudFieldMESH_VERSION 1

void
PointCloudMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), PointCloudFieldMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream,points_);

  stream.end_class();
}

const string
PointCloudMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "PointCloudMesh";
  return name;
}

void
PointCloudMesh::begin(PointCloudMesh::Node::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::end(PointCloudMesh::Node::iterator &itr) const
{
  itr = (unsigned)points_.size();
}

void
PointCloudMesh::size(PointCloudMesh::Node::size_type &s) const
{
  s = (unsigned)points_.size();
}

void
PointCloudMesh::begin(PointCloudMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::end(PointCloudMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::size(PointCloudMesh::Edge::size_type &s) const
{
  s = 0;
}

void
PointCloudMesh::begin(PointCloudMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::end(PointCloudMesh::Face::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::size(PointCloudMesh::Face::size_type &s) const
{
  s = 0;
}

void
PointCloudMesh::begin(PointCloudMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::end(PointCloudMesh::Cell::iterator &itr) const
{
  itr = 0;
}

void
PointCloudMesh::size(PointCloudMesh::Cell::size_type &s) const
{
  s = 0;
}



const TypeDescription*
PointCloudMesh::get_type_description() const
{
  return SCIRun::get_type_description((PointCloudMesh *)0);
}

const TypeDescription*
get_type_description(PointCloudMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(PointCloudMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("PointCloudMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun
