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
 *  FieldSetMatrixProperty.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;
using namespace std;

class FieldSetMatrixProperty : public Module {
public:
  FieldSetMatrixProperty(GuiContext*);

  virtual ~FieldSetMatrixProperty();

  virtual void execute();

private:
  GuiString     guimatrix1name_;
  GuiString     guimatrix2name_;
  GuiString     guimatrix3name_;
};


DECLARE_MAKER(FieldSetMatrixProperty)
  FieldSetMatrixProperty::FieldSetMatrixProperty(GuiContext* ctx)
    : Module("FieldSetMatrixProperty", ctx, Source, "FieldsProperty", "ModelCreation"),
      guimatrix1name_(get_ctx()->subVar("matrix1-name")),
      guimatrix2name_(get_ctx()->subVar("matrix2-name")),
      guimatrix3name_(get_ctx()->subVar("matrix3-name"))
{
}


FieldSetMatrixProperty::~FieldSetMatrixProperty()
{
}


void
FieldSetMatrixProperty::execute()
{
  const string matrix1name = guimatrix1name_.get();
  const string matrix2name = guimatrix2name_.get();
  const string matrix3name = guimatrix3name_.get();
    
  FieldHandle handle;
  if (!get_input_handle("Field", handle)) return;
  
  // Scan matrix input port 1
  MatrixHandle fhandle;
  if (get_input_handle("Matrix1", fhandle, false))
  {
    handle->set_property(matrix1name, fhandle, false);
  }

  if (get_input_handle("Matrix2", fhandle, false))
  {
    handle->set_property(matrix2name,fhandle,false);
  }

  if (get_input_handle("Matrix3", fhandle, false))
  {
    handle->set_property(matrix3name,fhandle,false);
  }
        
  // Now post the output
  handle->generation = handle->compute_new_generation();
  send_output_handle("Field", handle);
}

} // end namespace

