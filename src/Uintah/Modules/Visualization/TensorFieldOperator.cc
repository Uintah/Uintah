#include "TensorFieldOperator.h"
#include "TensorOperatorFunctors.h"
#include <math.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Datatypes/TensorField.h>
#include <Uintah/Datatypes/NCTensorField.h>
#include <Uintah/Datatypes/CCTensorField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGCC.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRGCC.h>
//#include <SCICore/Math/Mat.h>

namespace Uintah {
namespace Modules {
 
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;


extern "C" Module* make_TensorFieldOperator( const clString& id ) { 
  return scinew TensorFieldOperator( id );
}

template<class TensorField, class Field>
void initField(TensorField* tensorField, Field* field);

template<class TensorField, class ScalarField, class TensorOp >
void computeScalars(TensorField* tensorField, ScalarField* scalarField,
		    TensorOp op /* TensorOp should be a functor for
				    converting tensors scalars */ );

TensorFieldOperator::TensorFieldOperator(const clString& id)
  : Module("TensorFieldOperator",id,Source),
    tclOperation("operation", id, this),
    tclRow("row", id, this),
    tclColumn("column", id, this),
    tclPlaneSelect("planeSelect", id, this),
    tclDelta("delta", id, this),
    tclEigen2DCalcType("eigen2D-calc-type", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = scinew TensorFieldIPort(this, "TensorField");
  sfout = scinew ScalarFieldOPort(this, "ScalarField");

  // Add ports to the Module
  add_iport(in);
  add_oport(sfout);
}
  
void TensorFieldOperator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  TensorFieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }

  NCTensorField* pNCTF;
  CCTensorField* pCCTF;
  ScalarFieldRG* scalarField;
  //VectorFieldRG* vectorField;

  if (pNCTF = dynamic_cast<NCTensorField*>( hTF.get_rep() )) {
    scalarField = scinew ScalarFieldRG();
    performOperation(pNCTF, scalarField);
     
  }
  else if (pCCTF = dynamic_cast<CCTensorField*>( hTF.get_rep() )) {
    scalarField = scinew ScalarFieldRGCC();
    performOperation(pCCTF, scalarField);
  }
  else {
    // error -- node or cell centered tensor field expected but
    // not received.
    std::cerr<<"NCTensorfield or CCTEnsorfield expected\n";
    //    tcl_status.set("Not NC or CC Tensorfield");
    return;
  }

  sfout->send(scalarField);
}

template<class TensorField, class ScalarField>
void TensorFieldOperator::performOperation(TensorField* tensorField,
					   ScalarField* scalarField)
{
  initField(tensorField, scalarField);

  switch(tclOperation.get()) {
  case 0: // extract element
    computeScalars(tensorField, scalarField,
		   TensorElementExtractionOp(tclRow.get(), tclColumn.get()));
    break;
  case 1: // 2D eigen-value/vector
    if (tclEigen2DCalcType.get() == 0) {
      // e1 - e2
      int plane = tclPlaneSelect.get();
      if (plane == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYOp());
      else if (plane == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZOp());
      else
	computeScalars(tensorField, scalarField, Eigen2DYZOp());
    }
    else {
      // sin(e1 - e2) / delta
      int plane = tclPlaneSelect.get();
      double delta = tclDelta.get();
      if (plane == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYSinOp(delta));
      else if (plane == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZSinOp(delta));
      else
	computeScalars(tensorField, scalarField, Eigen2DYZSinOp(delta));
    }
    break;
  case 2: // pressure
    computeScalars(tensorField, scalarField, PressureOp());
    break;
  case 3: // equivalent stress 
    computeScalars(tensorField, scalarField, EquivalentStressOp());
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << tclOperation.get() << "\n";
  }
}

template<class TensorField, class Field>
void initField(TensorField* tensorField, Field* field)
{
  Point lb(0, 0, 0), ub(0, 0, 0);
  IntVector lowIndex, highIndex;

  tensorField->get_bounds(lb, ub);
  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);

  field->resize(highIndex.x(), highIndex.y(), highIndex.z());
  field->set_bounds(lb, ub);
}

template<class TensorField, class ScalarField, class TensorOp >
void computeScalars(TensorField* tensorField, ScalarField* scalarField,
		    TensorOp op /* TensorOp should be a functor for
				    converting tensors scalars */ )
{
  Matrix3 M;
  IntVector lowIndex, highIndex;

  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);
  
  for (int x = lowIndex.x(); x < highIndex.x(); x++) {
    for (int y = lowIndex.y(); y < highIndex.y(); y++) {
      for (int z = lowIndex.z(); z < highIndex.z(); z++) {
	scalarField->grid(x, y, z) = op(tensorField->grid(x, y, z));
      }
    }
  }  
}

}
}


