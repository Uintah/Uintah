#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <cstdlib>

using namespace Uintah;

void BoundCondFactory::create(ProblemSpecP& child,BoundCondBase* &bc, 
                              int& mat_id)

{
  map<string,string> bc_attr;
  child->getAttributes(bc_attr);
  
  
  // Check to see if "id" is defined
  if (bc_attr.find("id") == bc_attr.end()) 
    SCI_THROW(ProblemSetupException("id is not specified in the BCType tag", __FILE__, __LINE__));
  
  if (bc_attr["id"] != "all"){
    std::istringstream ss(bc_attr["id"]);
    ss >> mat_id;
  }else{
    mat_id = -1;  
  }

  //  std::cout << "mat_id = " << mat_id << std::endl;
  // Determine whether or not things are a scalar, Vector or a NoValue, i.e.
  // Symmetry

  double d_value;
  Vector v_value;

  if (child->get("value",d_value) != 0)
    bc = scinew BoundCond<double>(bc_attr["label"],bc_attr["var"],d_value);
  else if (child->get("value",v_value) != 0) {
    bc = scinew BoundCond<Vector>(bc_attr["label"],bc_attr["var"],v_value);
    // std::cout << "v_value = " << v_value << std::endl;
  }
  else
    bc = scinew BoundCond<NoValue>(bc_attr["label"],bc_attr["var"]);
  

}

