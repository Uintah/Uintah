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

//    File   : CreateMesh.h
//    Author : Michael Callahan
//    Date   : July 2001

#if !defined(CreateMesh_h)
#define CreateMesh_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Containers/StringUtil.h>

#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/PrismVolField.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TriSurfField.h>

namespace SCIRun {


class CreateMeshAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(ProgressReporter *m,
			      MatrixHandle elements,
			      MatrixHandle positions,
			      MatrixHandle normals,
			      Field::data_location data_at) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const string &basename,
					    const string &datatype);
};


template <class FIELD>
class CreateMeshAlgoT : public CreateMeshAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *m,
			      MatrixHandle elements,
			      MatrixHandle positions,
			      MatrixHandle normals,
			      Field::data_location data_at);
};


template <class FIELD>
FieldHandle
CreateMeshAlgoT<FIELD>::execute(ProgressReporter *mod,
				MatrixHandle elements,
				MatrixHandle positions,
				MatrixHandle normals,
				Field::data_location data_at)
{
  if (positions->ncols() < 3)
  {
    mod->error("Mesh Positions must contain at least 3 columns for position data.");
    return 0;
  }
  if (positions->ncols() > 3)
  {
    mod->remark("Mesh Positions contains unused columns, only first three are used.");
  }

  typename FIELD::mesh_handle_type mesh = scinew typename FIELD::mesh_type();

  int i, j;
  const int pnrows = positions->nrows();
  for (i = 0; i < pnrows; i++)
  {
    const Point p(positions->get(i, 0),
		  positions->get(i, 1),
		  positions->get(i, 2));
    mesh->add_point(p);
  }

  int ecount = 0;
  for (i = 0; i < elements->nrows(); i++)
  {
    typename FIELD::mesh_type::Node::array_type nodes;
    for (j = 0; j < elements->ncols(); j++)
    {
      int index = (int)elements->get(i, j);
      if (index < 0 || index >= pnrows)
      {
	if (ecount < 10)
	{
	  mod->error("Bad index found at " + to_string(i) + ", "
		     + to_string(j));
	}
	index = 0;
	ecount++;
      }
      nodes.push_back(index);
    }
    mesh->add_elem(nodes);
  }
  if (ecount >= 10)
  {
    mod->error("..." + to_string(ecount-9) + " additional bad indices found.");
  }
  
  FIELD *field = scinew FIELD(mesh, data_at);

  return FieldHandle(field);
}


} // end namespace SCIRun

#endif // CreateMesh_h
