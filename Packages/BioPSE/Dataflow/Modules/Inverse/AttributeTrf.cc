//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : AttributeTrf.cc
//    Author : yesim
//    Date   : Sat Feb  9 11:36:47 2002

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Algorithms/Geometry/SurfaceLaplacian.h>

#include <Packages/BioPSE/share/share.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Containers/Array2.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurfMesh.h>

#include <math.h>


namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE AttributeTrf : public Module {

public:

  // CONSTRUCTOR
  AttributeTrf(GuiContext *context);

  // DESTRUCTOR
  virtual ~AttributeTrf();

  virtual void execute();
};

DECLARE_MAKER(AttributeTrf)


// CONSTRUCTOR
AttributeTrf::AttributeTrf(GuiContext *context)
  : Module("AttributeTrf", context, Source, "Inverse", "BioPSE")
{
}

// DESTRUCTOR
AttributeTrf::~AttributeTrf(){
}

///////////////////////////////////////////////
// MODULE EXECUTION
///////////////////////////////////////////////
void AttributeTrf::execute(){

  FieldIPort *iportGeomF = (FieldIPort *)get_iport("InputFld");
  MatrixOPort *oportAttrib = (MatrixOPort *)get_oport("OutputMat");

  if (!iportGeomF) {
    error("Unable to initialize iport 'InputFld'.");
    return;
  }
  if (!oportAttrib) {
    error("Unable to initialize oport 'OutputMat'.");
    return;
  }

  // getting input field
  FieldHandle hFieldGeomF;
  if(!iportGeomF->get(hFieldGeomF) || !hFieldGeomF.get_rep()) { 
    error("Couldn't get handle to the Input Field.");
    return;
  }

  TriSurfMesh *tsm=dynamic_cast<TriSurfMesh *>(hFieldGeomF->mesh().get_rep());
  if (!tsm) {
    error("Input field was not a TriSurfField.");
    return;
  }
  oportAttrib->send(MatrixHandle(surfaceLaplacian(tsm)));
}

} // End namespace BioPSE
