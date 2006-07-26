/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <sci_defs/teem_defs.h>

#ifdef HAVE_TEEM
#include <teem/air.h>
#endif

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

#ifdef HAVE_TEEM
    if (!iss)
    {
      char ibuf[5];
      iss.clear();
      iss.get(ibuf,4);
      
      // Make sure the comparison is case insensitive.
      airToLower(ibuf);
      if (strncmp(ibuf,"nan",3)==0)
      {
        ptr[p] = static_cast<double>(AIR_NAN);
      }
      else if (strncmp(ibuf,"inf",3)==0)
      {
        ptr[p] = static_cast<double>(AIR_POS_INF);
      }
      else
      {
        iss.clear();
        iss.get(&(ibuf[3]),2);
        // Set the last character to null for airToLower.
        airToLower(ibuf);
        if (strncmp(ibuf,"-inf",4)==0)
        {
          ptr[p] = static_cast<double>(AIR_NEG_INF);
        }  	  	
        else
        {
          error("Matrix contains invalid information");
          return;
        }
      }
    }
#endif      
  }

  DenseMatrix *dmat = mat->dense();
  handle = dynamic_cast<Matrix *>(dmat);
  delete mat;
  
  send_output_handle("Matrix", handle);
}

} // End namespace SCIRun


