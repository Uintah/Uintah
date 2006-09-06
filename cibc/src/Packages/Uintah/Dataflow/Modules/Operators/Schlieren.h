#ifndef __OPERATORS_Schlieren_H__
#define __OPERATORS_Schlieren_H__

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Packages/Uintah/Dataflow/Modules/Operators/UnaryFieldOperator.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;


class SchlierenAlgo: public DynamicAlgoBase, public UnaryFieldOperator
{
public:

  virtual FieldHandle execute( FieldHandle density,
                               double dx, double dy, double dz) = 0;
  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
 };  

template<class FIELD>
class SchlierenAlgoT: public SchlierenAlgo
{
public:
  virtual FieldHandle execute(FieldHandle scalarfh,
                               double dx, double dy, double dz);
private:
  void computeSchlierenImage(FIELD* density,
                             double dx, double dy, double dz,
                             FIELD* output);
};

template<class FIELD >
FieldHandle
SchlierenAlgoT<FIELD>::execute(FieldHandle scalarfh,
                               double dx, double dy, double dz)
{
  FIELD *density = (FIELD *)(scalarfh.get_rep());
  typename FIELD::mesh_handle_type mh = density->get_typed_mesh();
  mh.detach();
  typename FIELD::mesh_type *mesh = mh.get_rep();

  FIELD *scalarField = scinew FIELD(mesh);

  computeSchlierenImage( density, dx, dy, dz, scalarField );

  return scalarField;
}

//______________________________________________________________________
//  This computes a schlieren image of the density field.
//  SI = sqrt(gx^2 + gy^2 + gz^2)
template<class FIELD >       
void
SchlierenAlgoT<FIELD>::computeSchlierenImage(FIELD* density, 
                                             double dx, double dy, double dz,
                                             FIELD* output)
{

  typedef typename FIELD::mesh_type MESH;

  initField(density, output);

  // only works with cell Centered data
  ASSERT( density->basis_order() == 0 );
  typename FIELD::mesh_handle_type sfmh =density->get_typed_mesh();
  typename FIELD::mesh_type *mesh =density->get_typed_mesh().get_rep();

  typename FIELD::mesh_type::Cell::iterator v_it;
  typename FIELD::mesh_type::Cell::iterator v_end;  
  sfmh->begin(v_it);
  sfmh->end(v_end);
  
  BBox mesh_boundary = density->mesh()->get_bounding_box();
  Point min, max;
  min = mesh_boundary.min(); 
  max = mesh_boundary.max();
  typename MESH::Cell::index_type min_index, max_index;
  mesh->get_cells(min_index, max_index, BBox(min, max));
  
//  debugging
//  cout << " dx " << dx << " dy " << dy << " dz " << dz << endl; 
//  cout << " mesh min: " << min << " max " << max << endl;
//  cout << " min " << min_index << " max " << max_index << endl;
//  cout << " v_begin " << v_it << " v_end " << v_end << endl;

  for( ; v_it != v_end; ++v_it){ 
    double schlieren = 0;    
    typename MESH::Cell::index_type c(mesh, v_it.i_,  v_it.j_, v_it.k_);

    // don't get near the edge of the computational domain
    if ( (c.i_ > min_index.i_  && c.j_ > min_index.j_    && c.k_ > min_index.k_ ) &&   
         (c.i_ < max_index.i_  && c.j_ < max_index.j_    && c.k_ < max_index.k_ ) ){   

      typename MESH::Cell::index_type R(mesh, c.i_+1, c.j_,   c.k_);
      typename MESH::Cell::index_type L(mesh, c.i_-1, c.j_,   c.k_);
      typename MESH::Cell::index_type T(mesh, c.i_,   c.j_+1, c.k_); 
      typename MESH::Cell::index_type B(mesh, c.i_,   c.j_-1, c.k_);
      typename MESH::Cell::index_type F(mesh, c.i_,   c.j_,   c.k_+1);
      typename MESH::Cell::index_type BK(mesh,c.i_,   c.j_,   c.k_-1);
     
      double gx = (density->fdata()[R] - density->fdata()[L])/2.0  * dx;
      double gy = (density->fdata()[T] - density->fdata()[B])/2.0  * dy;
      double gz = (density->fdata()[F] - density->fdata()[BK])/2.0 * dz;
 
      schlieren = sqrt(gx*gx + gy*gy + gz*gz);
   }
   output->fdata()[c] =schlieren;
  }
}

} // end namespace Uintah

#endif // __OPERATORS_Schlieren_H__

