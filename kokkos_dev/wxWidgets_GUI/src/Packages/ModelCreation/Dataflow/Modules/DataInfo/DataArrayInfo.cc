/*
 *  DataArrayInfo.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Containers/Handle.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace ModelCreation {

using namespace SCIRun;

class DataArrayInfo : public Module {
public:
  DataArrayInfo(GuiContext*);

  virtual ~DataArrayInfo();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString gui_matrixname_;
  GuiString gui_generation_;
  GuiString gui_typename_;
  GuiString gui_elements_;

  int generation_;

  void clear_vals();
  void update_input_attributes(MatrixHandle);

};


DECLARE_MAKER(DataArrayInfo)
DataArrayInfo::DataArrayInfo(GuiContext* ctx)
  : Module("DataArrayInfo", ctx, Source, "DataInfo", "ModelCreation"),
    gui_matrixname_(ctx->subVar("matrixname", false)),
    gui_generation_(ctx->subVar("generation", false)),
    gui_typename_(ctx->subVar("typename", false)),
    gui_elements_(ctx->subVar("elements", false)),
    generation_(-1)
{
  gui_matrixname_.set("---");
  gui_generation_.set("---");
  gui_typename_.set("---");
  gui_elements_.set("---");
}

DataArrayInfo::~DataArrayInfo(){
}

void DataArrayInfo::execute()
{
  // The input port (with data) is required.
  MatrixIPort *iport = (MatrixIPort*)get_iport("DataArray");
  MatrixHandle mh;
  if (!iport->get(mh) || !mh.get_rep())
  {
    clear_vals();
    generation_ = -1;
    return;
  }

  if (generation_ != mh.get_rep()->generation)
  {
    generation_ = mh.get_rep()->generation;
    update_input_attributes(mh);
  }
  
  MatrixOPort *oport;
  
  if (oport = dynamic_cast<MatrixOPort *>(get_oport("NumElements")))
  {
    MatrixHandle nrows = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if(nrows.get_rep() == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }
    double* dataptr = nrows->get_data_pointer();
    if (dataptr == 0)
    {
      error("Could not allocate enough memory for output matrix");
      return;
    }    
    dataptr[0] = static_cast<double>(mh->nrows());
    oport->send(nrows);
  }
}

void
 DataArrayInfo::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


void DataArrayInfo::clear_vals()
{
  gui_matrixname_.set("---");
  gui_generation_.set("---");
  gui_typename_.set("---");
  gui_elements_.set("---");
}

void DataArrayInfo::update_input_attributes(MatrixHandle m)
{
  string matrixname;
  if (m->get_property("name", matrixname))
  {
    gui_matrixname_.set(matrixname);
  }
  else
  {
    gui_matrixname_.set("--- Name Not Assigned ---");
  }

  gui_generation_.set(to_string(m->generation));

  // Set the typename.
  if (m->ncols() == 1)
  {
    gui_typename_.set("ScalarArray");
    gui_elements_.set(to_string(m->nrows()));  
  }
  else if (m->ncols() == 3)
  {
    gui_typename_.set("VectorArray");
    gui_elements_.set(to_string(m->nrows()));  
  }
  else if ((m->ncols() == 6)||(m->ncols() == 9))
  {
    gui_typename_.set("TensorArray");
    gui_elements_.set(to_string(m->nrows()));  
  }
  else
  {
    gui_typename_.set("--- invalid array type ---");
    gui_elements_.set(to_string(m->nrows()));  
  }
}

} // End namespace ModelCreation


