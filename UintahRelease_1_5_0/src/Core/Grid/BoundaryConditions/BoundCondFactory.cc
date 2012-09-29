/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <cstdlib>

using namespace Uintah;
using std::string;
using std::map;

void BoundCondFactory::create(ProblemSpecP& child,BoundCondBase* &bc, 
                              int& mat_id, const std::string face_label)

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

  ProblemSpecP valuePS = child->findBlock( "value" );

  if( valuePS != 0) { // Found <value> tag.    
    try {
      child->get( "value", d_value );
      bc = scinew BoundCond<double>( bc_attr["label"], bc_attr["var"], d_value, face_label );
    }
    catch( ... ) {
      // If there was an exception, then the 'value' was not a double... try to get a vector...
      child->get( "value", v_value );
      bc = scinew BoundCond<Vector>( bc_attr["label"], bc_attr["var"], v_value, face_label );
    }
  }
  else {
    bc = scinew BoundCond<NoValue>( bc_attr["label"], bc_attr["var"] );
  }
}

