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
 *  SetupFEMatrix.cc:  Setups the global finite element matrix
 *
 *  Written by:
 *   Ruth Nicholson Klepfer
 *   Department of Bioengineering
 *   University of Utah
 *   October 1994
 *
 *  Modified:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001    
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildFEMatrix.h>

#include <iostream>

using std::cerr;
using std::endl;

namespace BioPSE {

using namespace SCIRun;
typedef LockingHandle<TetVol<int> >    CondMeshHandle;

class SetupFEMatrix : public Module {
  
  //! Private data
  FieldIPort*        iportField_;
  MatrixOPort*       oportMtrx_;
 
  GuiInt             uiUseCond_;
  int                lastUseCond_;
  
  MatrixHandle       hGblMtrx_;
  int                gen_;

public:
  
  //! Constructor/Destructor
  SetupFEMatrix(const string& id);
  virtual ~SetupFEMatrix();

  //! Public methods
  virtual void execute();
};

extern "C" Module* make_SetupFEMatrix(const string& id) {
  return new SetupFEMatrix(id);
}

SetupFEMatrix::SetupFEMatrix(const string& id): 
  Module("SetupFEMatrix", id, Filter, "Forward", "BioPSE"), 
  uiUseCond_("UseCondTCL", id, this),
  lastUseCond_(1)
{
  gen_=-1;
  uiUseCond_.set(1);
}

SetupFEMatrix::~SetupFEMatrix(){
}

void SetupFEMatrix::execute(){
  
  iportField_ = (FieldIPort *)get_iport("Mesh");
  oportMtrx_ = (MatrixOPort *)get_oport("Stiffness Matrix");

  if (!iportField_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!oportMtrx_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle hField;
  if(!iportField_->get(hField)){
    msgStream_ << "Error: Cann't get field" << endl;
    return;
  }

  if (hField->generation == gen_ 
      && hGblMtrx_.get_rep() 
      && lastUseCond_==uiUseCond_.get()) {
    oportMtrx_->send(hGblMtrx_);
    return;
  }
  
  gen_ = hField->generation;
  CondMeshHandle hCondMesh;
  if (hField->get_type_name(0)=="TetVol" && hField->get_type_name(1)=="int"){
    
    hCondMesh = dynamic_cast<TetVol<int>* >(hField.get_rep());
    
    if (!hCondMesh.get_rep()){
      msgStream_ << "Error: Unable to cast to TetVol<int>*" << endl;
      return;
    }
  }
  else {
    msgStream_ << "Error: The mesh in conductivity tensor field is not of type TetVol<int>" << endl;
    return;
  }
  

  //! finding conductivity tensor lookup table
  vector<pair<string, Tensor> > tens;

  if (uiUseCond_.get()==1 && hCondMesh->get("conductivity_table", tens)){
    msgStream_ << "Using supplied conductivity tensors "  << endl;
  }
  else {
    msgStream_ << "Using identity conductivity tensors" << endl;
    pair<int,int> minmax;
    minmax.second=1;
    field_minmax(*(hCondMesh.get_rep()), minmax);
    tens.resize(minmax.second+1);
    vector<double> t(6);
    t[0] = t[3] = t[5] = 1;
    t[1] = t[2] = t[4] = 0;
    Tensor ten(t);
    for (unsigned int i = 0; i < tens.size(); i++)
    {
      tens[i] = pair<string, Tensor>(to_string((int)i), ten);
    }
  }
  
  lastUseCond_ = uiUseCond_.get();
  if(BuildFEMatrix::build_FEMatrix(hCondMesh, tens, hGblMtrx_)){
    msgStream_ << "Matrix is ready" << endl;
    msgStream_ << "Size: " << hGblMtrx_->nrows() << "-by-" << hGblMtrx_->ncols() << endl;
  };
  
  //! outputing
  oportMtrx_->send(hGblMtrx_);
}

} // End namespace BioPSE
