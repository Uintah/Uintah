/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef __FACECUTTINGPLANE_H_
#define __FACECUTTINGPLANE_H_

#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/Field.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h> 
#include <Core/Geometry/Transform.h>

using namespace SCIRun;
namespace Uintah {

class FaceCuttingPlaneAlgo: public DynamicAlgoBase
{
public:

  virtual bool get_value(FieldHandle fld, Point p, double& val) = 0;
  virtual bool get_bounds_and_dimensions(FieldHandle fld, int& nx, int& ny, 
                                         int& nz, BBox& bbox) = 0;
  virtual void set_transform(FieldHandle fld, Transform& t) = 0;
  
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
};

template <class FIELD>
class FaceCuttingPlaneAlgoT : public FaceCuttingPlaneAlgo
{
public:
  virtual bool get_value(FieldHandle fld, Point p, double& val);
  virtual bool get_bounds_and_dimensions(FieldHandle fld, int& nx, int& ny, 
                                         int& nz, BBox& bbox);
  virtual void set_transform(FieldHandle fld, Transform& t);
};


template<class FIELD> 
bool
FaceCuttingPlaneAlgoT<FIELD>::get_bounds_and_dimensions( FieldHandle fld, 
                                                  int& nx, int& ny, 
                                                  int& nz, BBox& bbox)
{
  const TypeDescription *td = 
    fld->get_type_description(Field::MESH_TD_E);
  if( td->get_name().find("LatVolMesh") == string::npos ){
    return false;
  }

  FIELD *field = (FIELD *)fld.get_rep();
  typename FIELD::mesh_handle_type mesh = field->get_typed_mesh();
  
  nx = mesh->get_ni();
  ny = mesh->get_nj();
  nz = mesh->get_nk();

  bbox = mesh->get_bounding_box();
  
  return true;
}

template<class FIELD> 
bool
FaceCuttingPlaneAlgoT<FIELD>::get_value(FieldHandle fld, Point p, double& val)
{

  FIELD *lvf = (FIELD *)(fld.get_rep());
  typename FIELD::mesh_type *mesh = lvf->get_typed_mesh().get_rep();

  typename FIELD::mesh_type::Node::index_type node;
  if( mesh->locate(node, p) ){
    val = lvf->fdata()[ node ];
    return true;
  } else {
    return false;
  }
}

template<class FIELD> 
void
FaceCuttingPlaneAlgoT<FIELD>::set_transform(FieldHandle fld, Transform& t)
{
 FIELD *field = (FIELD *)fld.get_rep();
  typename FIELD::mesh_handle_type mh = field->get_typed_mesh();
  
  mh->transform(t);
}

} // end namespace Uintah

#endif
