#ifndef SCIRun_InsertVoltageSource_H
#define SCIRun_InsertVoltageSource_H
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
