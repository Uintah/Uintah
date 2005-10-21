#ifndef __CUTTINGPLANE_H_
#define __CUTTINGPLANE_H_

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h> 


using namespace SCIRun;
namespace Uintah {

class CuttingPlaneAlgo: public DynamicAlgoBase
{
public:
  virtual bool get_dimensions(FieldHandle fld, int& nx, int& ny, int& nz) = 0;
  virtual bool get_gradient(FieldHandle texfld_, const Point& p, Vector& g)=0;
  
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
};

template <class FIELD>
class CuttingPlaneAlgoT: public CuttingPlaneAlgo
{
  virtual bool get_dimensions(FieldHandle fld, int& nx, int& ny, int& nz);
  virtual bool get_gradient(FieldHandle texfld_, const Point& p, Vector& g);
};


template <class FIELD>
bool
CuttingPlaneAlgoT<FIELD>::get_dimensions( FieldHandle fld, 
                                          int& nx, int& ny, int& nz)
{
  if( fld->mesh()->get_type_name(0) != "LatVolMesh" ) return false;

  FIELD *field = (FIELD *)fld.get_rep();
  typename FIELD::mesh_handle_type mesh = field->get_typed_mesh();
  
  nx = mesh->get_ni();
  ny = mesh->get_nj();
  nz = mesh->get_nk();
  
  return true;
}
template <class FIELD>
bool
CuttingPlaneAlgoT<FIELD>::get_gradient(FieldHandle fld, 
                                       const Point& p, Vector& g)
{
  if( fld->mesh()->get_type_name(0) != "LatVolMesh" ) return false;

  // To be implemented.
}


} // end namespace Uintah

#endif
