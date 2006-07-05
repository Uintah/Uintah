/*
 *  CreateMatrix.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

namespace SCIRun {

using namespace SCIRun;

class CreateMatrix : public Module {
public:
  CreateMatrix(GuiContext*);

  virtual ~CreateMatrix();

  virtual void execute();

private:
  GuiInt    nrows_;
  GuiInt    ncols_;
  GuiString data_;
  GuiString clabel_;
  GuiString rlabel_;
};


DECLARE_MAKER(CreateMatrix)
CreateMatrix::CreateMatrix(GuiContext* ctx)
  : Module("CreateMatrix", ctx, Source, "Math", "SCIRun"),
    nrows_(get_ctx()->subVar("rows"), 1),
    ncols_(get_ctx()->subVar("cols"), 1),
    data_(get_ctx()->subVar("data"), "{0.0}"),
    clabel_(get_ctx()->subVar("clabel"), "{0}"),
    rlabel_(get_ctx()->subVar("rlabel"), "{0}")
{
}


CreateMatrix::~CreateMatrix()
{
}


void
CreateMatrix::execute()
{
  MatrixHandle handle;
  get_gui()->execute(get_id() + " update_matrixdata");
  
  int nrows = nrows_.get();
  int ncols = ncols_.get();
  std::string data = data_.get();
  
  DenseColMajMatrix *mat = scinew DenseColMajMatrix(nrows,ncols);
  for (size_t p=0;p<data.size();p++)
  { 
    if ((data[p] == '}')||(data[p] == '{')) data[p] = ' ';
  }
  
  double *ptr = mat->get_data_pointer();
  
  std::istringstream iss(data);
  for (int p = 0; p < (nrows*ncols); p++)
  {
      iss >> ptr[p];
  }

  DenseMatrix *dmat = mat->dense();
  handle = dynamic_cast<Matrix *>(dmat);
  delete mat;
  
  MatrixOPort *oport;
  if (!(oport = dynamic_cast<MatrixOPort *>(get_oport(0))))
  {
    error("Cannot find ouput port");
    return;
  }
  
  oport->send_and_dereference(handle);
}

} // End namespace SCIRun


