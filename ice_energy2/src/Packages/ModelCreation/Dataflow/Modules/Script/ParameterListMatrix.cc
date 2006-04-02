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

/*
 * FILE: ParameterListMatrix.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 17 SEP 2005
 */ 

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class ParameterListMatrix : public Module {
public:
  ParameterListMatrix(GuiContext*);

  virtual ~ParameterListMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
private:

  GuiString             guimatrixname_;
  GuiString             guimatrices_;  
};


DECLARE_MAKER(ParameterListMatrix)
ParameterListMatrix::ParameterListMatrix(GuiContext* ctx)
  : Module("ParameterListMatrix", ctx, Source, "Script", "ModelCreation"),
    guimatrixname_(get_ctx()->subVar("matrix-name")),
    guimatrices_(get_ctx()->subVar("matrix-selection"))
{
}

ParameterListMatrix::~ParameterListMatrix()
{
}

void ParameterListMatrix::execute()
{
  string matrixname = guimatrixname_.get();
  string matrixlist;
        
  BundleHandle handle;
  BundleIPort  *iport;
  BundleOPort  *oport;
  
  MatrixHandle fhandle;
  MatrixIPort *ifport;
  MatrixOPort *ofport;
        
  if(!(iport = static_cast<BundleIPort *>(get_iport("ParameterList"))))
  {
    error("Cannot not find ParameterList input port");
    return;
  }

  if(!(ifport = static_cast<MatrixIPort *>(get_iport("Matrix"))))
  {
    error("Cannot not find Matrix input port");
    return;
  }

  // If no input bundle is found, create a new one

  iport->get(handle);
  if (handle.get_rep() == 0)
  {   
    handle = dynamic_cast<Bundle *>(scinew Bundle());
  }

  ifport->get(fhandle);
  if (fhandle.get_rep() != 0)
  {
    handle = handle->clone();
    handle->setMatrix(matrixname,fhandle);
    fhandle = 0;
  }

  // Update the GUI with all the matrix names
  // So the can select which one he or she wants
  // to extract
  
  size_t nummatrices = handle->numMatrices();
  for (size_t p = 0; p < nummatrices; p++)
  {
    matrixlist += "{" + handle->getMatrixName(p) + "} ";
  }
  guimatrices_.set(matrixlist);
  get_ctx()->reset();


  if (!(ofport = static_cast<MatrixOPort *>(get_oport("Matrix"))))
  {
    error("Could not find Matrix output port");
    return; 
  }
  if (!(oport = static_cast<BundleOPort *>(get_oport("ParameterList"))))
  {
    error("Could not find ParameterList output port");
    return; 
  }
 
  if (handle->isMatrix(matrixname))
  {
    fhandle = handle->getMatrix(matrixname);
    ofport->send(fhandle);
  }        
          
  oport->send(handle);

}

void
 ParameterListMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


