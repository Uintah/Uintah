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

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>

namespace ModelCreation {

using namespace SCIRun;
using namespace std;

class FieldGetMatrixProperty : public Module {
public:
  FieldGetMatrixProperty(GuiContext*);

  virtual void execute();
  
private:
  GuiString             guimatrix1name_;
  GuiString             guimatrix2name_;
  GuiString             guimatrix3name_;
  GuiString             guimatrixs_;
};


DECLARE_MAKER(FieldGetMatrixProperty)
  FieldGetMatrixProperty::FieldGetMatrixProperty(GuiContext* ctx)
    : Module("FieldGetMatrixProperty", ctx, Source, "FieldsProperty", "ModelCreation"),
      guimatrix1name_(get_ctx()->subVar("matrix1-name")),
      guimatrix2name_(get_ctx()->subVar("matrix2-name")),
      guimatrix3name_(get_ctx()->subVar("matrix3-name")),
      guimatrixs_(get_ctx()->subVar("matrix-selection"))
{
}


void
FieldGetMatrixProperty::execute()
{
  string matrix1name = guimatrix1name_.get();
  string matrix2name = guimatrix2name_.get();
  string matrix3name = guimatrix3name_.get();
  string matrixlist;
        
  FieldHandle handle;
  FieldIPort  *iport;
  MatrixHandle fhandle;
        
  if(!(iport = static_cast<FieldIPort *>(get_input_port("Field"))))
  {
    error("Could not find 'Field' input port");
    return;
  }

  if (!(iport->get(handle)))
  {   
    warning("No field connected to the input port");
    return;
  }

  if (handle.get_rep() == 0)
  {   
    warning("Input field is empty");
    return;
  }

  size_t nprop = handle->nproperties();

  for (size_t p=0;p<nprop;p++)
  {
    handle->get_property(handle->get_property_name(p),fhandle);
    if (fhandle.get_rep()) matrixlist += "{" + handle->get_property_name(p) + "} ";
  }

  guimatrixs_.set(matrixlist);
  get_ctx()->reset();
 
  if (handle->is_property(matrix1name))
  {
    handle->get_property(matrix1name, fhandle);
    send_output_handle("Matrix1", fhandle);
  }

  if (handle->is_property(matrix2name))
  {
    handle->get_property(matrix2name,fhandle);
    send_output_handle("Matrix2", fhandle);
  }
        
  if (handle->is_property(matrix3name))
  {
    handle->get_property(matrix3name,fhandle);
    send_output_handle("Matrix3", fhandle);
  }        
}


} // end namespace
