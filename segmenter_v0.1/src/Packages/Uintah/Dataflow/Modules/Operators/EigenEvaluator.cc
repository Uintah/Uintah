#include <math.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Packages/Uintah/Core/Math/Matrix3.h>


#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace SCIRun;
namespace Uintah {
using std::string;

class EigenEvaluator: public Module {
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LVMesh::handle_type                 LVMeshHandle;
  typedef HexTrilinearLgn<double>             FDdoubleBasis;
  typedef ConstantBasis<double>               CFDdoubleBasis; 
  typedef HexTrilinearLgn<Vector>             FDVectorBasis;
  typedef ConstantBasis<Vector>               CFDVectorBasis;
  typedef HexTrilinearLgn<Matrix3>             FDMatrix3Basis;
  typedef ConstantBasis<Matrix3>               CFDMatrix3Basis;

  typedef GenericField<LVMesh, CFDdoubleBasis, FData3d<double, LVMesh> > CDField;
  typedef GenericField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> > LDField;
  typedef GenericField<LVMesh, CFDVectorBasis, FData3d<Vector, LVMesh> > CVField;
  typedef GenericField<LVMesh, FDVectorBasis,  FData3d<Vector, LVMesh> > LVField;
  typedef GenericField<LVMesh, CFDMatrix3Basis, FData3d<Matrix3, LVMesh> > CMField;
  typedef GenericField<LVMesh, FDMatrix3Basis,  FData3d<Matrix3, LVMesh> > LMField;
  EigenEvaluator(GuiContext* ctx);
  virtual ~EigenEvaluator() {}
    
  virtual void execute(void);
    
private:
  //    TCLstring tcl_status;
  GuiInt guiEigenSelect;
  FieldIPort *in;

  FieldOPort *sfout; // for eigen values
  FieldOPort *vfout; // for eigen vectors
};

} // end namespace Uintah

using namespace Uintah;

  DECLARE_MAKER(EigenEvaluator)

template<class TensorField, class VectorField, class ScalarField>
void computeGridEigens(TensorField* tensorField,
		       ScalarField* eValueField, VectorField* eVectorField,
		       int chosenEValue);
 

EigenEvaluator::EigenEvaluator(GuiContext* ctx)
  : Module("EigenEvaluator",ctx,Source, "Operators", "Uintah"),
    guiEigenSelect(get_ctx()->subVar("eigenSelect"))
    //    tcl_status(get_ctx()->subVar("tcl_status")),
{
}
  
void EigenEvaluator::execute(void) {
  //  tcl_status.set("Calling EigenEvaluator!"); 

  in = (FieldIPort *) get_iport("Tensor Field");
  sfout = (FieldOPort *) get_oport("Eigenvalue Field");
  vfout = (FieldOPort *) get_oport("Eigenvector Field");
  

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"EigenEvaluator::execute(void) Didn't get a handle\n";
    return;
  }

  const TypeDescription *td = hTF->get_type_description(Field::MESH_TD_E);
  if ( td->get_name().find("Matrix3") == string::npos ){
    std::cerr<<"Input is not a Tensor field\n";
    return;
  }

  CDField *cdefield = 0;
  LDField *ldefield = 0;
  CVField *cvefield = 0;
  LVField *lvefield = 0;

  if( CMField *tensorField = 
      dynamic_cast<CMField *>(hTF.get_rep())){
    LVMeshHandle mh = tensorField->get_typed_mesh();
    mh.detach();
    cdefield = scinew CDField( mh );
    mh.detach();
    cvefield = scinew CVField( mh );
    computeGridEigens( tensorField, cdefield, cvefield, guiEigenSelect.get());
  } else if(LMField *tensorField = 
      dynamic_cast<LMField *>(hTF.get_rep())){
    LVMeshHandle mh = tensorField->get_typed_mesh();
    mh.detach();
    ldefield = scinew LDField( mh );
    mh.detach();
    lvefield = scinew LVField( mh );
    computeGridEigens( tensorField, ldefield, lvefield, guiEigenSelect.get());
  }
  
  if( cdefield && cvefield ){
    sfout->send(cdefield);
    vfout->send(cvefield);
  } else if( ldefield && lvefield){
    sfout->send(ldefield);
    vfout->send(lvefield);
  }
}

template<class TensorField, class VectorField, class ScalarField>
void computeGridEigens(TensorField* tensorField,
		       ScalarField* eValueField, VectorField* eVectorField,
		       int chosenEValue)
{
  ASSERT( tensorField->basis_order() == 0 ||
	  tensorField->basis_order() == 1 );
  typename TensorField::mesh_handle_type tmh = tensorField->get_typed_mesh();
  typename ScalarField::mesh_handle_type smh = eValueField->get_typed_mesh();
  typename VectorField::mesh_handle_type vmh = eVectorField->get_typed_mesh();

  BBox box;
  box = tmh->get_bounding_box();

  //resize the geometry
  smh->set_ni(tmh->get_ni());
  smh->set_nj(tmh->get_nj());
  smh->set_nk(tmh->get_nk());
  smh->set_transform(tmh->get_transform());
  vmh->set_ni(tmh->get_ni());
  vmh->set_nj(tmh->get_nj());
  vmh->set_nk(tmh->get_nk());
  smh->set_transform(tmh->get_transform());
  //resize the data storage
  eValueField->resize_fdata();
  eVectorField->resize_fdata();


  int num_eigen_values;
  Matrix3 M;
  double e[3];
  std::vector<Vector> eigenVectors;
  
  if( tensorField->basis_order() == 0){
    typename TensorField::mesh_type::Cell::iterator t_it; tmh->begin(t_it);
    typename ScalarField::mesh_type::Cell::iterator s_it; smh->begin(s_it);
    typename VectorField::mesh_type::Cell::iterator v_it; vmh->begin(v_it);
    typename TensorField::mesh_type::Cell::iterator t_end; tmh->end(t_end);
    
    for( ; t_it != t_end; ++t_it, ++s_it, ++v_it){
      M = tensorField->fdata()[*t_it];
      num_eigen_values = M.getEigenValues(e[0], e[1], e[2]);
      if (num_eigen_values <= chosenEValue) {
	eValueField->fdata()[*s_it] = 0;
	eVectorField->fdata()[*v_it] = Vector(0,0,0);
      } else {
	eValueField->fdata()[*s_it] = e[chosenEValue];
	eigenVectors = M.getEigenVectors(e[chosenEValue], e[0]);
	if (eigenVectors.size() != 1) {
	  eVectorField->fdata()[*v_it] = Vector(0, 0, 0);
	} else {
	  eVectorField->fdata()[*v_it] = eigenVectors[0].normal();
	}
      }
    }
  } else {
    typename TensorField::mesh_type::Node::iterator t_it; tmh->begin(t_it);
    typename ScalarField::mesh_type::Node::iterator s_it; smh->begin(s_it);
    typename VectorField::mesh_type::Node::iterator v_it; vmh->begin(v_it);
    typename TensorField::mesh_type::Node::iterator t_end; tmh->end(t_end);
    
    for( ; t_it != t_end; ++t_it, ++s_it, ++v_it){
      M = tensorField->fdata()[*t_it];
      num_eigen_values = M.getEigenValues(e[0], e[1], e[2]);
      if (num_eigen_values <= chosenEValue) {
	eValueField->fdata()[*s_it] = 0;
	eVectorField->fdata()[*v_it] = Vector(0,0,0);
      } else {
	eValueField->fdata()[*s_it] = e[chosenEValue];
	eigenVectors = M.getEigenVectors(e[chosenEValue], e[0]);
	if (eigenVectors.size() != 1) {
	  eVectorField->fdata()[*v_it] = Vector(0, 0, 0);
	} else {
	  eVectorField->fdata()[*v_it] = eigenVectors[0].normal();
	}
      }
    }
  }    
}
  




