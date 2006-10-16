/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

using namespace SCIRun;

class ParameterListMatrix : public Module {
public:
  ParameterListMatrix(GuiContext*);
  virtual void execute();
  
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

void ParameterListMatrix::execute()
{
  BundleHandle bundle;
  MatrixHandle matrix;
  
  get_input_handle("ParameterList",bundle,false);
  get_input_handle("Matrix",matrix,false);

  if (inputs_changed_ || guimatrixname_.changed()  || !oport_cached("ParameterList") || !oport_cached("Matrix"))
  {
    std::string matrixname = guimatrixname_.get();
    std::string matrixlist;
        
    if (bundle.get_rep() == 0)
    {   
      bundle = scinew Bundle();
    }

    if (matrix.get_rep() != 0)
    {
      bundle = bundle->clone();
      bundle->setMatrix(matrixname,matrix);
      matrix = 0;
    }

    // Update the GUI with all the matrix names
    // So the can select which one he or she wants
    // to extract
    
    size_t nummatrices = bundle->numMatrices();
    for (size_t p = 0; p < nummatrices; p++)
    {
      matrixlist += "{" + bundle->getMatrixName(p) + "} ";
    }
    guimatrices_.set(matrixlist);
    get_ctx()->reset();

    if (bundle->isMatrix(matrixname))
    {
      matrix = bundle->getMatrix(matrixname);
      send_output_handle("Matrix",matrix,false);
    }        

    send_output_handle("ParameterList",bundle,false);
  }
}

} // End namespace ModelCreation


