#include "TensorElementExtractor.h"
#include <math.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Datatypes/TensorField.h>
#include <Uintah/Datatypes/NCTensorField.h>
#include <Uintah/Datatypes/CCTensorField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGCC.h>
//#include <SCICore/Math/Mat.h>

namespace Uintah {
namespace Modules {
 
using namespace SCICore::Containers;
using namespace PSECore::Dataflow;


extern "C" Module* make_TensorElementExtractor( const clString& id ) { 
  return scinew TensorElementExtractor( id );
}

template<class TensorField, class ScalarField>
void extractElements(TensorField* tensorField, ScalarField* elementValues,
		     int row, int column);
 

TensorElementExtractor::TensorElementExtractor(const clString& id)
  : Module("TensorElementExtractor",id,Source),
    tclRow("row", id, this),
    tclColumn("column", id, this)
    //    tcl_status("tcl_status", id, this),
{
  // Create Ports
  in = new TensorFieldIPort(this, "TensorField");
  sfout = new ScalarFieldOPort(this, "EigenValueField");

  // Add ports to the Module
  add_iport(in);
  add_oport(sfout);
}
  
void TensorElementExtractor::execute(void) {
  //  tcl_status.set("Calling EigenEvaluator!"); 
  TensorFieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Didn't get a handle\n";
    return;
  }


  NCTensorField* pNCTF;
  CCTensorField* pCCTF;
  ScalarFieldRG* eElementValues;
  
  if (pNCTF = dynamic_cast<NCTensorField*>( hTF.get_rep() )) {
    eElementValues = scinew ScalarFieldRG();
    extractElements(pNCTF, eElementValues, tclRow.get(), tclColumn.get());
  }
  else if (pCCTF = dynamic_cast<CCTensorField*>( hTF.get_rep() )) {
    eElementValues = scinew ScalarFieldRGCC();
    extractElements(pCCTF, eElementValues, tclRow.get(), tclColumn.get());
  }
  else {
    // error -- node or cell centered tensor field expected but
    // not received.
    std::cerr<<"NCTensorfield or CCTEnsorfield expected\n";
    //    tcl_status.set("Not NC or CC Tensorfield");
    return;
  }

  sfout->send(eElementValues);
}

template<class TensorField, class ScalarField>
void extractElements(TensorField* tensorField, ScalarField* elementValues,
		     int row, int column)
{
  ASSERTL3(row > 0 && row <= 3 && column > 0 && column <= 3);
  Point lb(0, 0, 0), ub(0, 0, 0);
  IntVector lowIndex, highIndex;

  tensorField->get_bounds(lb, ub);
  tensorField->GetLevel()->getIndexRange(lowIndex, highIndex);

  elementValues->resize(highIndex.x(), highIndex.y(), highIndex.z());
  elementValues->set_bounds(lb, ub);

  for (int x = lowIndex.x(); x < highIndex.x(); x++) {
    for (int y = lowIndex.y(); y < highIndex.y(); y++) {
      for (int z = lowIndex.z(); z < highIndex.z(); z++) {
	Matrix3 M = tensorField->grid(x, y, z);
	elementValues->grid(x, y, z) = M(row, column);
      }
    }
  }
  
}
  
}
}


