#ifndef SCIRun_InsertVoltageSource_H
#define SCIRun_InsertVoltageSource_H
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


#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/Point.h>
#include <Core/Algorithms/Visualization/RenderField.h>

namespace SCIRun {

// just get the first point from the mesh
class InsertVoltageSourceGetPtBase : public DynamicAlgoBase
{
public:
  virtual Point execute(MeshHandle src) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class InsertVoltageSourceGetPt : public InsertVoltageSourceGetPtBase
{
public:
  //! virtual interface. 
  virtual Point execute(MeshHandle src);
};

template <class MESH>
Point 
InsertVoltageSourceGetPt<MESH>::execute(MeshHandle mesh_h)
{
  typedef typename MESH::Node::iterator node_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  node_iter_type ni; mesh->begin(ni);
  Point p(0,0,0);
  mesh->get_center(p, *ni);
  return p;
}

// find all of the data_at locations and return them and their values
class InsertVoltageSourceGetPtsAndValsBase : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle fld, vector<Point> &pts, vector<double> &vals) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fld_td,
					    const TypeDescription *loc_td);
};


template <class Fld, class Loc>
class InsertVoltageSourceGetPtsAndVals : public InsertVoltageSourceGetPtsAndValsBase
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle fld, vector<Point> &pts, vector<double> &vals);
};

template <class Fld, class Loc>
void 
InsertVoltageSourceGetPtsAndVals<Fld, Loc>::execute(FieldHandle fld_h, 
						    vector<Point> &pts, 
						    vector<double> &vals)
{
  Fld *fld = dynamic_cast<Fld*>(fld_h.get_rep());
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();
  typename Loc::iterator itr, itr_end;
  typename Fld::value_type tmp;
  mesh->begin(itr);
  mesh->end(itr_end);
  Point p;
  double val;
  while (itr != itr_end) {
    mesh->get_center(p, *itr);
    pts.push_back(p);
    fld->value(tmp, *itr);
    to_double(tmp, val);
    vals.push_back(val);
    ++itr;
  }
}
} // End namespace BioPSE

#endif
