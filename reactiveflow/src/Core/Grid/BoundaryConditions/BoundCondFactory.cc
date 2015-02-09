/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <cstdlib>

using namespace Uintah;
using std::string;
using std::map;

//______________________________________________________________________
void BoundCondFactory::create(ProblemSpecP& child,BoundCondBase* &bc, 
                              int& mat_id, const std::string face_label)

{
  map<string,string> bc_attr;
  child->getAttributes(bc_attr);
//
//  // Check to see if "id" is defined
//  if (bc_attr.find("id") == bc_attr.end()) 
//    SCI_THROW(ProblemSetupException("id is not specified in the BCType tag", __FILE__, __LINE__));
//  
//  if (bc_attr["id"] != "all"){
//    std::istringstream ss(bc_attr["id"]);
//    ss >> mat_id;
//  }else{
//    mat_id = -1;  
//  }

  //  std::cout << "mat_id = " << mat_id << std::endl;
  // Determine whether or not things are a scalar, Vector or a NoValue, i.e.
  // Symmetry
  int    i_value;
  double d_value;
  Vector v_value;
  string s_value = "none";
  
  string valAttribute;
  bool attrPS = child->getAttribute("value",valAttribute);
  ProblemSpecP valuePS = child->findBlock( "value" );
  
  if (valuePS && attrPS) {
    // the user specified BOTH <value> </value> AND value="" attribute
    SCI_THROW(ProblemSetupException("Error: It looks like you specified two values for BC " + bc_attr["label"] + ". This is not allowed! You could only specify either a value attribute or a <value> node. Please revise your input file.", __FILE__, __LINE__));
  }

//
// TSAAD: ACHTUNG!
// Due to the fact that different Uintah components deal with parsing the boundary-conditions spec
// in their own way, it was found that removing the following warning was best to avoid compaints
// from several developers. Bear in mind that removing this warning places the burden of BC proper-parsing
// on component developers. I disagree with that, given the limited resource to refactor, this was the best choice that could be made.
//
//  if (!valuePS && !attrPS) {
//    if (bc_attr["label"] != "symmetry" || bc_attr["label"] != "zeroNeumann") { // specific handing for ICE and MPM since they parse BCs differently
//      proc0cout << "WARNING: It looks like you specified no value for BC " + bc_attr["label"] + ". This may be okay if your component allows you to for certain types of boundaries such as symmetry and zeroNeumann.\n";
//    }
//  }

  if (attrPS) {
    ProblemSpec::InputType theInputType = child->getInputType(valAttribute);
    switch (theInputType) {
      case ProblemSpec::NUMBER_TYPE:
      {
        if( bc_attr["type"] == "int" ){ // integer ONLY if the tag 'type = "int"' is added
          child->getAttribute( "value", i_value);
          bc = scinew BoundCond<int> ( bc_attr["label"], bc_attr["var"], i_value, face_label, BoundCondBase::INT_TYPE );
        }else{ // double (default)
          child->getAttribute( "value", d_value );
          bc = scinew BoundCond<double>( bc_attr["label"], bc_attr["var"], d_value, face_label, BoundCondBase::DOUBLE_TYPE );
        }
      }
        break;
      case ProblemSpec::VECTOR_TYPE:
        child->getAttribute( "value", v_value );
        bc = scinew BoundCond<Vector>( bc_attr["label"], bc_attr["var"], v_value, face_label, BoundCondBase::VECTOR_TYPE );
        break;
      case ProblemSpec::STRING_TYPE:
        bc = scinew BoundCond<std::string>( bc_attr["label"], bc_attr["var"], valAttribute, face_label, BoundCondBase::STRING_TYPE );
        break;
      case ProblemSpec::UNKNOWN_TYPE:
      default:
        bc = scinew BoundCond<NoValue>( bc_attr["label"], bc_attr["var"] );
        break;
    }
  } else if( valuePS ) { // Found <value> tag.
    child->get( "value", s_value );
    ProblemSpec::InputType theInputType = child->getInputType(s_value);
    
    switch (theInputType) {
      case ProblemSpec::NUMBER_TYPE:
        if( bc_attr["type"] == "int" ){                  // integer ONLY if the tag 'type = "int"' is added
          child->get( "value", i_value);
          bc = scinew BoundCond<int> ( bc_attr["label"], bc_attr["var"], i_value, face_label, BoundCondBase::INT_TYPE );
        }else{                                           // double (default)
          child->get( "value", d_value );
          bc = scinew BoundCond<double>( bc_attr["label"], bc_attr["var"], d_value, face_label, BoundCondBase::DOUBLE_TYPE );
        }
        break;
      case ProblemSpec::VECTOR_TYPE:
        child->get( "value", v_value );
        bc = scinew BoundCond<Vector>( bc_attr["label"], bc_attr["var"], v_value, face_label, BoundCondBase::VECTOR_TYPE );
        break;
      case ProblemSpec::STRING_TYPE:
        bc = scinew BoundCond<std::string>( bc_attr["label"], bc_attr["var"], s_value, face_label, BoundCondBase::STRING_TYPE );
        break;
      case ProblemSpec::UNKNOWN_TYPE:
      default:
        bc = scinew BoundCond<NoValue>( bc_attr["label"], bc_attr["var"] );
        break;
    }
  } else {
    bc = scinew BoundCond<NoValue>( bc_attr["label"], bc_attr["var"] );
  }
}

