#include "EigenEvaluator.h"
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


extern "C" Module* make_EigenEvaluator( const clString& id ) { 
  return scinew EigenEvaluator( id );
}

template<class TensorField, class VectorField, class ScalarField>
void computeGridEigens(TensorField* tensorField,
		       ScalarField* eValueField, VectorField* eVectorField,
		       int chosenEValue);
 

EigenEvaluator::EigenEvaluator(const clString& id)
  : Module("EigenEvaluator",id,Source),
    tclEigenSelect("eigenSelect", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = new TensorFieldIPort(this, "TensorField");
  sfout = new ScalarFieldOPort(this, "EigenValueField");
  vfout = new VectorFieldOPort(this, "EigenVectorField");

  // Add ports to the Module
  add_iport(in);
  add_oport(sfout);
  add_oport(vfout);
}
  
void EigenEvaluator::execute(void) {
  //  tcl_status.set("Calling EigenEvaluator!"); 
  TensorFieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }


  NCTensorField* pNCTF;
  CCTensorField* pCCTF;
  ScalarFieldRG* eValueField;
  VectorFieldRG* eVectorField;

  
  if (pNCTF = dynamic_cast<NCTensorField*>( hTF.get_rep() )) {
    eValueField = scinew ScalarFieldRG();
    eVectorField = scinew VectorFieldRG();
    computeGridEigens(pNCTF, eValueField, eVectorField, tclEigenSelect.get());
  }
  else if (pCCTF = dynamic_cast<CCTensorField*>( hTF.get_rep() )) {
    eValueField = scinew ScalarFieldRGCC();
    eVectorField = scinew VectorFieldRGCC();
    computeGridEigens(pCCTF, eValueField, eVectorField, tclEigenSelect.get());
  }
  else {
    // error -- node or cell centered tensor field expected but
    // not received.
    std::cerr<<"NCTensorfield or CCTEnsorfield expected\n";
    //    tcl_status.set("Not NC or CC Tensorfield");
    return;
  }

  sfout->send(eValueField);
  vfout->send(eVectorField);  
}

template<class TensorField, class VectorField, class ScalarField>
void computeGridEigens(TensorField* tensorField,
		       ScalarField* eValueField, VectorField* eVectorField,
		       int chosenEValue)
{
  Point lb(0, 0, 0), ub(0, 0, 0);
  IntVector lowIndex, highIndex;

  tensorField->computeHighLowIndices();
  tensorField->get_bounds(lb, ub);
  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);

  eValueField->resize(highIndex.x(), highIndex.y(), highIndex.z());
  eValueField->set_bounds(lb, ub);
  eVectorField->resize(highIndex.x(), highIndex.y(), highIndex.z());
  eVectorField->set_bounds(lb, ub);

  int num_eigen_values;
  Matrix3 M;
  double e[3];
  std::vector<Vector> eigenVectors;
  
  for (int x = lowIndex.x(); x < highIndex.x(); x++) {
    for (int y = lowIndex.y(); y < highIndex.y(); y++) {
      for (int z = lowIndex.z(); z < highIndex.z(); z++) {
	M = tensorField->grid(x, y, z);
	num_eigen_values = M.getEigenValues(e[0], e[1], e[2]);
	if (num_eigen_values != 3) {
	  eValueField->grid(x, y, z) = 0;
	  eVectorField->grid(x, y, z) = Vector(0, 0, 0);
	}
	else {
	  eValueField->grid(x, y, z) = e[chosenEValue];
	  eigenVectors = M.getEigenVectors(e[chosenEValue]);
	  if (eigenVectors.size() != 1) {
	    eVectorField->grid(x, y, z) = Vector(0, 0, 0);
	  }
	  else {
	    eVectorField->grid(x, y, z) = eigenVectors[0].normal();
	  }
	}
      }
    }
  }
  
}
  
}
}


