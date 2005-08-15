#include <Packages/Uintah/Dataflow/Modules/Operators/ScalarFieldNormalize.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>

#include <math.h>

using namespace SCIRun;
using namespace Uintah;
 
DECLARE_MAKER(ScalarFieldNormalize)

ScalarFieldNormalize::ScalarFieldNormalize(GuiContext* ctx)
  : Module("ScalarFieldNormalize",ctx,Source, "Operators", "Uintah"),
    xIndex_(ctx->subVar("xIndex")), yIndex_(ctx->subVar("yIndex")), zIndex_(ctx->subVar("zIndex"))
{
}
  
void ScalarFieldNormalize::execute(void) 
{
  in    = (FieldIPort *) get_iport("Scalar Field");
  sfout = (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"ScalarFieldNormalize::execute(void) Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "double" ){
    std::cerr<<"Input is not a Scalar field\n";
    return;
  }

  FieldHandle fh = 0;
  if( LatVolField<double> *input =
      dynamic_cast<LatVolField<double>*>(hTF.get_rep())) {
    LatVolField<double>  *output = 0;  

    output = scinew LatVolField<double>(hTF->basis_order());

    normalizeScalarField(input, output);
    fh = output;
  }
  if( fh.get_rep() != 0 ){
    sfout->send(fh);
  }
}


//_____________________________________________________________
template<class ScalarField>       
void  
ScalarFieldNormalize::normalizeScalarField(ScalarField* input,
                                           ScalarField* output)
{
  initField(input, output);
  int x = xIndex_.get() + 1;  // get user input
  int y = yIndex_.get() + 1;  // Note that CC index space differs from LatVolMesh
  int z = zIndex_.get() + 1;  // index space by (1,1,1). 
  
  // only CC data
  ASSERT( input->basis_order() == 0 );
  ASSERT( output->basis_order() == 0 );
  typename ScalarField::mesh_handle_type sfmh =input->get_typed_mesh();
  typename ScalarField::mesh_type *mesh =input->get_typed_mesh().get_rep();

  typename ScalarField::mesh_type::Cell::iterator v_it;
  typename ScalarField::mesh_type::Cell::iterator v_end;  
  sfmh->begin(v_it);
  sfmh->end(v_end);
  
  BBox mesh_boundary = input->mesh()->get_bounding_box();
  Point min, max;
  min = mesh_boundary.min(); 
  max = mesh_boundary.max();
  LatVolMesh::Cell::index_type min_index, max_index;
  mesh->get_cells(min_index, max_index, BBox(min, max));
  
  LatVolMesh::Cell::index_type probePt(mesh, x,y,z);
  double normalizingValue = input->fdata()[probePt];
    
  //cout << "ScalarFieldNormalize: LatVolMesh index space: min " 
  //     << min_index << " max " << max_index << endl;
  cout << "ScalarFieldNormalize: probePt in LatVolMesh index space" 
       << probePt << " normalizing value " << normalizingValue << endl;
  
  for( ; v_it != v_end; ++v_it){
    output->fdata()[*v_it] =input->fdata()[*v_it]/normalizingValue;
  }
}




