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
