
#include <Packages/Uintah/Dataflow/Modules/Operators/Schlieren.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>

#include <math.h>

using namespace SCIRun;
using namespace Uintah;

DECLARE_MAKER(Schlieren)

Schlieren::Schlieren(GuiContext* ctx)
  : Module("Schlieren",ctx,Source, "Operators", "Uintah"),
    dx_(ctx->subVar("dx")), dy_(ctx->subVar("dy")), dz_(ctx->subVar("dz"))
{
}
  
void
Schlieren::execute(void) 
{
  in_    = (FieldIPort *) get_iport("Scalar Field");
  sfout_ = (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  // bullet proofing
  if(!in_->get(hTF)){
    std::cerr<<"Schlieren::execute(void) Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "double" ){
    std::cerr<<"Input is not a Scalar field\n";
    return;
  }

  FieldHandle fh = 0;
  
  if( LatVolField<double> *scalarField =
      dynamic_cast<LatVolField<double>*>(hTF.get_rep())) {
    
    LatVolField<double>  *output = 0;  
    output = scinew LatVolField<double>(hTF->basis_order());
    
    // compute the image
    computeSchlierenImage(scalarField,output); 
    
    fh = output;
  }
  if( fh.get_rep() != 0 )
    sfout_->send(fh);
}
//______________________________________________________________________
//  This computes a schlieren image of the density field.
//  SI = sqrt(gx^2 + gy^2 + gz^2)
template<class ScalarField>       
void
Schlieren::computeSchlierenImage(ScalarField* density, ScalarField* output)
{

  initField(density, output);
  // Grab the values from the GUI at the beginning of the loop (so
  // that the user doesn't change them in the middle.
  double dx = dx_.get();
  double dy = dy_.get();
  double dz = dz_.get();

  // only works with cell Centered data
  ASSERT( density->basis_order() == 0 );
  typename ScalarField::mesh_handle_type sfmh =density->get_typed_mesh();
  typename ScalarField::mesh_type *mesh =density->get_typed_mesh().get_rep();

  typename ScalarField::mesh_type::Cell::iterator v_it;
  typename ScalarField::mesh_type::Cell::iterator v_end;  
  sfmh->begin(v_it);
  sfmh->end(v_end);
  
  BBox mesh_boundary = density->mesh()->get_bounding_box();
  Point min, max;
  min = mesh_boundary.min(); 
  max = mesh_boundary.max();
  LatVolMesh::Cell::index_type min_index, max_index;
  mesh->get_cells(min_index, max_index, BBox(min, max));
  
//  debugging
//  cout << " dx " << dx << " dy " << dy << " dz " << dz << endl; 
//  cout << " mesh min: " << min << " max " << max << endl;
//  cout << " min " << min_index << " max " << max_index << endl;
//  cout << " v_begin " << v_it << " v_end " << v_end << endl;

  for( ; v_it != v_end; ++v_it){ 
    double schlieren = 0;    
    LatVolMesh::Cell::index_type c(mesh, v_it.i_,  v_it.j_, v_it.k_);

    // don't get near the edge of the computational domain
    if ( (c.i_ > min_index.i_  && c.j_ > min_index.j_    && c.k_ > min_index.k_ ) &&   
         (c.i_ < max_index.i_  && c.j_ < max_index.j_    && c.k_ < max_index.k_ ) ){   

      LatVolMesh::Cell::index_type R(mesh, c.i_+1, c.j_,   c.k_);
      LatVolMesh::Cell::index_type L(mesh, c.i_-1, c.j_,   c.k_);
      LatVolMesh::Cell::index_type T(mesh, c.i_,   c.j_+1, c.k_); 
      LatVolMesh::Cell::index_type B(mesh, c.i_,   c.j_-1, c.k_);
      LatVolMesh::Cell::index_type F(mesh, c.i_,   c.j_,   c.k_+1);
      LatVolMesh::Cell::index_type BK(mesh,c.i_,   c.j_,   c.k_-1);
     
      double gx = (density->fdata()[R] - density->fdata()[L])/2.0  * dx;
      double gy = (density->fdata()[T] - density->fdata()[B])/2.0  * dy;
      double gz = (density->fdata()[F] - density->fdata()[BK])/2.0 * dz;
 
      schlieren = sqrt(gx*gx + gy*gy + gz*gz);
   }
   output->fdata()[c] =schlieren;
  }
}
