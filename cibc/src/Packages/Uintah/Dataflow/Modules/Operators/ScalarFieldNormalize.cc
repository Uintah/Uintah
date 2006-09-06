
#include <math.h>

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Packages/Uintah/Dataflow/Modules/Operators/UnaryFieldOperator.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>



#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace SCIRun;

namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;

class ScalarFieldNormalize: public Module, public UnaryFieldOperator {
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LVMesh::handle_type                 LVMeshHandle;
  typedef HexTrilinearLgn<double>             FDdoubleBasis;
  typedef ConstantBasis<double>               CFDdoubleBasis; 

  typedef GenericField<LVMesh, CFDdoubleBasis, FData3d<double, LVMesh> > CDField;
  typedef GenericField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> > LDField;

  ScalarFieldNormalize(GuiContext* ctx);
  virtual ~ScalarFieldNormalize() {}
    
  virtual void execute(void);
    
protected:
  template<class ScalarField>       
  void normalizeScalarField(ScalarField* input,ScalarField* output);
    
private:
  GuiInt xIndex_, yIndex_, zIndex_;

  FieldIPort *in;
  FieldOPort *sfout;
};
} // end namespace Uintah

using namespace Uintah;
 
DECLARE_MAKER(ScalarFieldNormalize)

ScalarFieldNormalize::ScalarFieldNormalize(GuiContext* ctx)
  : Module("ScalarFieldNormalize",ctx,Source, "Operators", "Uintah"),
    xIndex_(get_ctx()->subVar("xIndex")), yIndex_(get_ctx()->subVar("yIndex")), zIndex_(get_ctx()->subVar("zIndex"))
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
  } else if ( !hTF->query_scalar_interface(this).get_rep() ){
    error("Input is not a Scalar field");
    return;
  }

  FieldHandle fh = 0;
  if( CDField *input = dynamic_cast<CDField *>(hTF.get_rep())) {
    LVMeshHandle mh = input->get_typed_mesh();
    mh.detach();
    CDField *output = scinew CDField( mh );
    normalizeScalarField(input, output);
    fh = output;
//   } else if( LDField *input = dynamic_cast<LDField *>(hTF.get_rep())) {
//     LVMeshHandle mh = input->get_typed_mesh();
//     mh.detach();
//     LDField *output = scinew LDField( mh );
//     normalizeScalarField(input, output);
//     fh = output;
  } else {
    error("Normalization only works on Cell Centered (basis->order() == 0) double fields\n");
    return;
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
  LVMesh::Cell::index_type min_index, max_index;
  mesh->get_cells(min_index, max_index, BBox(min, max));
  
  LVMesh::Cell::index_type probePt(mesh, x,y,z);
  double normalizingValue = input->fdata()[probePt];
    
  //cout << "ScalarFieldNormalize: LatVolMesh index space: min " 
  //     << min_index << " max " << max_index << endl;
  cout << "ScalarFieldNormalize: probePt in LatVolMesh index space" 
       << probePt << " normalizing value " << normalizingValue << endl;
  
  for( ; v_it != v_end; ++v_it){
    output->fdata()[*v_it] =input->fdata()[*v_it]/normalizingValue;
  }
}




