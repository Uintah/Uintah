#include "InPlaneEigenEvaluator.h"
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


extern "C" Module* make_InPlaneEigenEvaluator( const clString& id ) { 
  return scinew InPlaneEigenEvaluator( id );
}

template<class TensorField, class ScalarField>
void initDataField(TensorField* tensorField, ScalarField* eDataField);

template<class TensorField, class ScalarField>
void computeGridEigenDiff(TensorField* tensorField, ScalarField* eValueField, 
			  int chosenPlane);


template<class TensorField, class ScalarField>
void computeGridSinEigenDiff(TensorField* tensorField, ScalarField* eDataField,
			     int chosenPlane, double delta);

typedef int (Matrix3::*pmfnPlaneEigenFunc)(double& e1, double& e2) const;
pmfnPlaneEigenFunc planeEigenValueFuncs[3];
  
InPlaneEigenEvaluator::InPlaneEigenEvaluator(const clString& id)
  : Module("InPlaneEigenEvaluator",id,Source),
    tclPlaneSelect("planeSelect", id, this),
    tclCalculationType("calcType", id, this),
    tclDelta("delta", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = new TensorFieldIPort(this, "TensorField");
  sfout = new ScalarFieldOPort(this, "EigenDataField");

  // Add ports to the Module
  add_iport(in);
  add_oport(sfout);

  planeEigenValueFuncs[0] = Matrix3::getYZEigenValues;
  planeEigenValueFuncs[1] = Matrix3::getXZEigenValues;
  planeEigenValueFuncs[2] = Matrix3::getXYEigenValues;
}
  
void InPlaneEigenEvaluator::execute(void) {
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  TensorFieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }


  NCTensorField* pNCTF;
  CCTensorField* pCCTF;
  ScalarFieldRG* eDataField;

  if (pNCTF = dynamic_cast<NCTensorField*>( hTF.get_rep() )) {
    eDataField = scinew ScalarFieldRG();
    initDataField(pNCTF, eDataField);
    if (tclCalculationType.get() == 0)
      computeGridEigenDiff(pNCTF, eDataField, tclPlaneSelect.get());
    else
      computeGridSinEigenDiff(pNCTF, eDataField, tclPlaneSelect.get(),
			      tclDelta.get());
  }
  else if (pCCTF = dynamic_cast<CCTensorField*>( hTF.get_rep() )) {
    eDataField = scinew ScalarFieldRGCC();
    if (tclCalculationType.get() == 0)   
      computeGridEigenDiff(pCCTF, eDataField, tclPlaneSelect.get());
    else
      computeGridSinEigenDiff(pCCTF, eDataField, tclPlaneSelect.get(),
			      tclDelta.get());
  }
  else {
    // error -- node or cell centered tensor field expected but
    // not received.
    std::cerr<<"NCTensorfield or CCTEnsorfield expected\n";
    //    tcl_status.set("Not NC or CC Tensorfield");
    return;
  }

  sfout->send(eDataField);
}

template<class TensorField, class ScalarField>
void initDataField(TensorField* tensorField, ScalarField* eDataField)
{
  Point lb(0, 0, 0), ub(0, 0, 0);
  IntVector lowIndex, highIndex;

  tensorField->get_bounds(lb, ub);
  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);

  eDataField->resize(highIndex.x(), highIndex.y(), highIndex.z());
  eDataField->set_bounds(lb, ub);
}

template<class TensorField, class ScalarField>
void computeGridEigenDiff(TensorField* tensorField, ScalarField* eDataField, 
			  int chosenPlane)
{
  int num_eigen_values;
  Matrix3 M;
  double e1, e2;
  IntVector lowIndex, highIndex;

  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);
  
  for (int x = lowIndex.x(); x < highIndex.x(); x++) {
    for (int y = lowIndex.y(); y < highIndex.y(); y++) {
      for (int z = lowIndex.z(); z < highIndex.z(); z++) {
	M = tensorField->grid(x, y, z);
	num_eigen_values = (M.*planeEigenValueFuncs[chosenPlane])(e1, e2);
	if (num_eigen_values != 2)
	  // There are either two equivalent eigen values or they are
	  // imaginary numbers.  Either case, just use 0 as the diff.
	  eDataField->grid(x, y, z) = 0;
	else
	  eDataField->grid(x, y, z) = e1 - e2; // e1 > e2
      }
    }
  }
  
}

template<class TensorField, class ScalarField>
void computeGridSinEigenDiff(TensorField* tensorField, ScalarField* eDataField,
			     int chosenPlane, double delta)
{
  int num_eigen_values;
  Matrix3 M;
  double e1, e2;
  IntVector lowIndex, highIndex;

  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);
  
  for (int x = lowIndex.x(); x < highIndex.x(); x++) {
    for (int y = lowIndex.y(); y < highIndex.y(); y++) {
      for (int z = lowIndex.z(); z < highIndex.z(); z++) {
	M = tensorField->grid(x, y, z);
	num_eigen_values = (M.*planeEigenValueFuncs[chosenPlane])(e1, e2);
	if (num_eigen_values != 2)
	  // There are either two equivalent eigen values or they are
	  // imaginary numbers.  Either case, just use 0 as the diff.
	  eDataField->grid(x, y, z) = 0;
	else
	  eDataField->grid(x, y, z) = sin((e1 - e2) / delta); // e1 > e2
      }
    }
  }
  
}

}
}


