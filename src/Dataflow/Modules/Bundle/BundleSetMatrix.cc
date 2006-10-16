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
#include <Dataflow/Network/Ports/BundlePort.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>

using namespace SCIRun;

class BundleSetMatrix : public Module {
public:
  BundleSetMatrix(GuiContext*);
  virtual void execute();

private:
  GuiString     guimatrix1name_;
  GuiString     guimatrix2name_;
  GuiString     guimatrix3name_;
  GuiString     guibundlename_;
};


DECLARE_MAKER(BundleSetMatrix)

BundleSetMatrix::BundleSetMatrix(GuiContext* ctx)
  : Module("BundleSetMatrix", ctx, Filter, "Bundle", "SCIRun"),
    guimatrix1name_(get_ctx()->subVar("matrix1-name"), "matrix1"),
    guimatrix2name_(get_ctx()->subVar("matrix2-name"), "matrix2"),
    guimatrix3name_(get_ctx()->subVar("matrix3-name"), "matrix3"),
    guibundlename_(get_ctx()->subVar("bundlename"), "")
{
}

void BundleSetMatrix::execute()
{
  BundleHandle  handle;
  MatrixHandle matrix1, matrix2, matrix3;

  get_input_handle("bundle",handle,false);
  get_input_handle("matrix1",matrix1,false);
  get_input_handle("matrix2",matrix2,false);
  get_input_handle("matrix3",matrix3,false);
  
  if (inputs_changed_ || guimatrix1name_.changed() || guimatrix2name_.changed() ||
      guimatrix3name_.changed() || guibundlename_.changed() || !oport_cached("bundle"))
  {
  
    std::string matrix1name = guimatrix1name_.get();
    std::string matrix2name = guimatrix2name_.get();
    std::string matrix3name = guimatrix3name_.get();
    std::string bundlename = guibundlename_.get();

    if (handle.get_rep())
    {
      handle.detach();
    }
    else
    {
      handle = scinew Bundle();
      if (handle.get_rep() == 0)
      {
        error("Could not allocate new bundle");
        return;
      }
    }
                
    if (matrix1.get_rep()) handle->setMatrix(matrix1name,matrix1);
    if (matrix2.get_rep()) handle->setMatrix(matrix2name,matrix2);
    if (matrix3.get_rep()) handle->setMatrix(matrix3name,matrix3);
    if (bundlename != "")
    {
      handle->set_property("name",bundlename,false);
    }

    send_output_handle("bundle",handle,false);
  }

}

