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
 *  GeomQMesh.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Geom/GeomQMesh.h>

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <iostream>
using std::ostream;

namespace SCIRun {

Persistent* make_GeomQMesh()
{
    return scinew GeomQMesh(0,0);
}

PersistentTypeID GeomQMesh::type_id("GeomQMesh", "GeomObj", make_GeomQMesh);

GeomQMesh::GeomQMesh(int nr, int nc)
:nrows(nr),ncols(nc)
{
  pts.resize(nr*nc*3);
  nrmls.resize(nr*nc*3);
  clrs.resize(nr*nc);
}

GeomQMesh::GeomQMesh(const GeomQMesh& copy)
: GeomObj(copy)
{
}

GeomQMesh::~GeomQMesh()
{
}

void GeomQMesh::add(int x, int y, Point& p, Vector& v, Color& c)
{
  int index3 = y*nrows*3 + x*3;
  int index  = y*nrows + x;

  pts[index3 + 0] = p.x();
  pts[index3 + 1] = p.y();
  pts[index3 + 2] = p.z();

  nrmls[index3 + 0] = v.x();
  nrmls[index3 + 1] = v.y();
  nrmls[index3 + 2] = v.z();

  clrs[index] = Colorub(c);
}

void GeomQMesh::get_bounds(BBox& bb)
{
    for(int i=0;i<pts.size()/3;i++){
	Point pp(pts[i*3 + 0],pts[i*3 + 1],pts[i*3 + 2]);
	bb.extend(pp);
    }
}

GeomObj* GeomQMesh::clone()
{
    return scinew GeomQMesh(*this);
}

#define GeomQMesh_VERSION 2

void GeomQMesh::io(Piostream&)
{

}    

} // End namespace SCIRun


