/*
 *  ElectrodeManager.cc:
 *
 *  Written by:
 *   lkreda
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

//#include <Core/Datatypes/TetVolField.h>
//#include <Core/Datatypes/TriSurfField.h>
//#include <Core/Datatypes/PointCloudField.h>
//#include <Dataflow/Modules/Fields/FieldInfo.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
//#include <Dataflow/Widgets/BoxWidget.h>
//#include <Core/Malloc/Allocator.h>
//#include <Core/Math/MinMax.h>
//#include <Core/Math/Trig.h>

#include <Core/GuiInterface/GuiVar.h>
#include <iostream>



#include <Packages/BioPSE/share/share.h>

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE ElectrodeManager : public Module {
  //! Private data

  //! Output port
  MatrixOPort*  electrodeParams_;
  MatrixOPort*  currPattIndicies_;

public:
  GuiInt modelTCL_;
  GuiInt numElTCL_;
  GuiDouble lengthElTCL_;
  GuiInt startNodeIdxTCL_;

  ElectrodeManager(GuiContext*);
  virtual ~ElectrodeManager();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(ElectrodeManager)

ElectrodeManager::ElectrodeManager(GuiContext* ctx)
  : Module("ElectrodeManager", ctx, Source, "Forward", "BioPSE"),
    modelTCL_(ctx->subVar("modelTCL")),
    numElTCL_(ctx->subVar("numElTCL")),
    lengthElTCL_(ctx->subVar("lengthElTCL")),
    startNodeIdxTCL_(ctx->subVar("startNodeIdxTCL"))

{
}

ElectrodeManager::~ElectrodeManager(){
}

void
 ElectrodeManager::execute()
{
  electrodeParams_ = (MatrixOPort *)get_oport("Electrode Parameters");
  currPattIndicies_ = (MatrixOPort *)get_oport("Current Pattern Index Vector");

  if (!electrodeParams_) {
    error("Unable to initialize oport 'Electrode Parameters'.");
    return;
  }

  if (!currPattIndicies_) {
    error("Unable to initialize oport 'Current Pattern Index Vector'.");
    return;
  }

  unsigned int model = modelTCL_.get();
  unsigned int numEl = Max(numElTCL_.get(), 0);
  double lengthEl = Max(lengthElTCL_.get(), 0.0);
  unsigned int startNodeIndex = Max(startNodeIdxTCL_.get(), 0);

  ColumnMatrix* elParams;
  elParams = scinew ColumnMatrix(4);

  if (model==0)
  {
    (*elParams)[0] = 0;
  }
  else
  {
    (*elParams)[0] = 1;  // gap model
  }
  (*elParams)[1]= (double) numEl;
  (*elParams)[2]= lengthEl;
  (*elParams)[3]= startNodeIndex;
 

  // There are numEl-1 unique current patterns
  // Current pattern index is 1-based
  ColumnMatrix* currPattIndicies;
  currPattIndicies = scinew ColumnMatrix(numEl-1);
  for (int i = 0; i < numEl-1; i++)
  {
    (*currPattIndicies)[i] = i + 1;
  }

  //! Sending result
  electrodeParams_->send(MatrixHandle(elParams)); 
  currPattIndicies_->send(MatrixHandle(currPattIndicies));

}

void
 ElectrodeManager::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace BioPSE


