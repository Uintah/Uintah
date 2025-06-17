/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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
// Note there are two ways to specify the value
//
//   1)  <BCType            label="pressure" var="Dirichlet" value="0.0"/>
//
//   2)  <BCType id = "0"   label = "Pressure"     var = "Neumann">
//                              <value> 0.0 </value>
//       </BCType>


void BoundCondFactory::create(ProblemSpecP  & BCType_ps,
                              BoundCondBase * &bc,
                              int           & mat_id,
                              const std::string face_label)

{
  map<string,string> bc_attr;
  BCType_ps->getAttributes(bc_attr);

  std::string label = bc_attr["label"];
  std::string var   = bc_attr["var"];

  int    i_value;
  double d_value;
  Vector v_value;
  string s_value = "none";

  string valAttribute;
  bool hasValueAttribute = BCType_ps->getAttribute("value",valAttribute);
  ProblemSpecP value_ps  = BCType_ps->findBlock( "value" );
  

  //__________________________________
  // bulletproofing.
  // BOTH <value> XXX </value> AND value="XXX" attribute
  // were specified
  if (value_ps && hasValueAttribute) {
    SCI_THROW(ProblemSetupException("Error: Two values for the BC " + label + " were specified. This is not allowed! " +
                                    " You can only specify a value attribute or a <value> node. Please revise your input file.",
                                     __FILE__, __LINE__));
  }

// TSAAD: ACHTUNG!
// Due to the fact that different Uintah components deal with parsing the boundary-conditions spec
// in their own way, it was found that removing the following warning was best to avoid compaints
// from several developers. Bear in mind that removing this warning places the burden of BC proper-parsing
// on component developers. I disagree with that, given the limited resource to refactor, this was the best choice that could be made.
//
//  if (!value_ps && !hasValueAttribute) {
//    if (bc_attr["label"] != "symmetry" || bc_attr["label"] != "zeroNeumann") { // specific handing for ICE and MPM since they parse BCs differently
//      proc0cout << "WARNING: It looks like you specified no value for BC " + bc_attr["label"] + ". This may be okay if your component allows you to for certain types of boundaries such as symmetry and zeroNeumann.\n";
//    }
//  }

  if ( hasValueAttribute ) {
    ProblemSpec::InputType valueType = BCType_ps->getInputType(valAttribute);

    switch (valueType) {
      case ProblemSpec::NUMBER_TYPE:
      {
        if( bc_attr["type"] == "int" ){                 // integer ONLY if the tag 'type = "int"' is added
          BCType_ps->getAttribute( "value", i_value);
          bc = scinew BoundCond<int> ( label, var, i_value, face_label, BoundCondBase::INT_TYPE );
        }
        else{                                           // double (default)
          BCType_ps->getAttribute( "value", d_value );
          bc = scinew BoundCond<double>( label, var, d_value, face_label, BoundCondBase::DOUBLE_TYPE );
        }
      }
        break;

      case ProblemSpec::VECTOR_TYPE:
        BCType_ps->getAttribute( "value", v_value );
        bc = scinew BoundCond<Vector>( label, var, v_value, face_label, BoundCondBase::VECTOR_TYPE );
        break;

      case ProblemSpec::STRING_TYPE:
        bc = scinew BoundCond<std::string>( label, var, valAttribute, face_label, BoundCondBase::STRING_TYPE );
        break;

      case ProblemSpec::UNKNOWN_TYPE:
      default:
        bc = scinew BoundCond<NoValue>( label, var );
        break;
    }
  }
  else if( value_ps ) { // Found <value> tag.
    BCType_ps->get( "value", s_value );

    ProblemSpec::InputType valueType = BCType_ps->getInputType( s_value );
    switch (valueType) {
      case ProblemSpec::NUMBER_TYPE:
        if( bc_attr["type"] == "int" ){                 // integer ONLY if the tag 'type = "int"' is added
          BCType_ps->get( "value", i_value);
          bc = scinew BoundCond<int> ( label, var, i_value, face_label, BoundCondBase::INT_TYPE );
        }
        else{                                          // double (default)
          BCType_ps->get( "value", d_value );
          bc = scinew BoundCond<double>( label, var, d_value, face_label, BoundCondBase::DOUBLE_TYPE );
        }
        break;

      case ProblemSpec::VECTOR_TYPE:
        BCType_ps->get( "value", v_value );
        bc = scinew BoundCond<Vector>( label, var, v_value, face_label, BoundCondBase::VECTOR_TYPE );
        break;

      case ProblemSpec::STRING_TYPE:
        bc = scinew BoundCond<std::string>( label, var, s_value, face_label, BoundCondBase::STRING_TYPE );
        break;

      case ProblemSpec::UNKNOWN_TYPE:
      default:
        bc = scinew BoundCond<NoValue>( label, var );
        break;
    }
  }
  else {
    bc = scinew BoundCond<NoValue>( label, var );
  }
}


//______________________________________________________________________
//                                                      Are these used?  --Todd
void BoundCondFactory::customBC(BoundCondBase* &bc,
    int mat_id, const std::string face_label, double value ,const std::string label,const std::string var){
      bc = scinew BoundCond<double>( label, var, value, face_label, BoundCondBase::DOUBLE_TYPE );
}

void BoundCondFactory::customBC(BoundCondBase* &bc,
    int mat_id, const std::string face_label,const Vector value ,const std::string label,const std::string var){
      bc = scinew BoundCond<Vector>( label, var, value, face_label, BoundCondBase::VECTOR_TYPE );
}

void BoundCondFactory::customBC(BoundCondBase* &bc,
    int mat_id, const std::string face_label,const std::string value ,const std::string label,const std::string var){
      bc = scinew BoundCond<std::string>( label, var, value, face_label, BoundCondBase::STRING_TYPE );
}
