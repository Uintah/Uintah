/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SetupBEMatrix.cc: constructs matrix Zbh to relate potentials on surfaces in 
 *                    boundary value problems
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December, 2000
 *   
 *   Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurf.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildBEMatrix.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <math.h>

#include <map>
#include <iostream>
#include <string>
#include <fstream>

namespace BioPSE {

using namespace SCIRun;


#define NUMZERO 10e-13
#define PI 3.141592653589738

// -------------------------------------------------------------------------------
class SetupBEMatrix : public Module {
  
  //! Input port
  FieldIPort*        iportSurfInn_;
  FieldIPort*        iportSurfOut_;
  
  //! Output ports
  MatrixOPort*       oportMatrix_;
  FieldOPort*        oportSurfOut_;
 
  MatrixHandle       hZbh_;
  ColumnMatrix       phiOut_;

  //! Old field generation
  int genIn_;
  int genOut_;
  
public:
  
  //! Constructor
  SetupBEMatrix(const string& id);
  
  //! Destructor  
  virtual ~SetupBEMatrix();

  virtual void execute();
};

//////////
// Module maker
extern "C" Module* make_SetupBEMatrix(const string& id) {
  return new SetupBEMatrix(id);
}

// -------------------------------------------------------------------------------
//////////
// Constructor/Destructor

SetupBEMatrix::SetupBEMatrix(const string& id): 
  Module("SetupBEMatrix", id, Source)
{
  // Create the input ports
  iportSurfInn_ = scinew FieldIPort(this, "Inner Surface", FieldIPort::Atomic);
  add_iport(iportSurfInn_);

  iportSurfOut_ = scinew FieldIPort(this, "Outer Surface", FieldIPort::Atomic);
  add_iport(iportSurfOut_);
  
  // Create the output ports
  oportMatrix_ = scinew MatrixOPort(this, "Zbh Matrix", MatrixIPort::Atomic);
  add_oport(oportMatrix_);

  oportSurfOut_ = scinew FieldOPort(this, "Outer Surf with Pots", FieldIPort::Atomic);
  add_oport(oportSurfOut_);

  genIn_= -1;
  genOut_= -1;
}

SetupBEMatrix::~SetupBEMatrix(){
}

// -------------------------------------------------------------------------------
//////////
// Module execution
void SetupBEMatrix::execute(){
  
  //! getting input fields
  FieldHandle hFieldInn;
  FieldHandle hFieldOut;
   
  if(!iportSurfInn_->get(hFieldInn)) { 
    msgStream_ << "Couldn't get handle to the inner surface. Returning." << endl;
    return;
  }
  
  if(!iportSurfOut_->get(hFieldOut)) { 
    msgStream_ << "Couldn't get handle to the outer surface. Returning." << endl;
    return;
  }

  
  //! TODO: precission checking
  if (hFieldInn->generation!=genIn_ || hFieldOut->generation!=genOut_ || !hZbh_.get_rep()){
    genIn_ = hFieldInn->generation;
    genOut_ = hFieldOut->generation;

    TriSurfMeshHandle  hSurfInn;
    TriSurfMeshHandle  hSurfOut;
    TriSurf<double>* pIn;
    TriSurf<double>* pOut;

    //! checking type consistency
    if (hFieldInn->get_type_name(0)=="TriSurf" && hFieldInn->get_type_name(1)=="double" ){
      pIn = dynamic_cast<TriSurf<double>*>(hFieldInn.get_rep());
      hSurfInn = pIn->get_typed_mesh();
    }
    else {
      msgStream_ << "Inner Surf is not of type TriSurf<double>" << endl;
      return;
    }
    
    if (hFieldOut->get_type_name(0)=="TriSurf" && hFieldInn->get_type_name(1)=="double"  ){
      pOut = dynamic_cast<TriSurf<double>*>(hFieldOut.get_rep());
      hSurfOut = pOut->get_typed_mesh();
    }
    else {
      msgStream_ << "Outer Surf is not of type TriSurf<double>" << endl;
      return;
    }
    
    BuildBEMatrix::build_BEMatrix(hSurfInn, hSurfOut, hZbh_, 2);
    
    if (!hZbh_.get_rep()){
      msgStream_ << "Unable to construct BE matrix " << endl;
      return;
    }
    
    vector<double>& pinn = pIn->fdata(); 
    if (pinn.size()==(unsigned int)pIn->get_typed_mesh()->nodes_size()){
      ColumnMatrix phiInn(pinn.size());
      phiOut_.resize(pOut->get_typed_mesh()->nodes_size());
      
      for (unsigned int i=0; i<pinn.size(); ++i){
	phiInn[i] = pinn[i];
      }
      int f, m;
      hZbh_->mult(phiInn, phiOut_, f, m);
    }
  }
  else {
    msgStream_ << "Field inputs are old. Resending stored matrix..." << endl;
  }

  TriSurf<double>* pOut2 = dynamic_cast<TriSurf<double>*>(hFieldOut.get_rep());
  
  if (pOut2){
    vector<double>& pout = pOut2->fdata();
    pout.resize(phiOut_.nrows()); 
    for (unsigned int i=0; i<pout.size(); ++i){
      pout[i] = phiOut_[i];
    }
  }

  // -- sending handles to cloned objects
  oportMatrix_->send(MatrixHandle(hZbh_->clone()));
  oportSurfOut_->send(FieldHandle(pOut2));
}

} // end namespace BioPSE
