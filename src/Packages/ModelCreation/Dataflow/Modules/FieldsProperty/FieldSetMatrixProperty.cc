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

  virtual void tcl_command(GuiArgs&, void*);
  
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

FieldSetMatrixProperty::~FieldSetMatrixProperty(){
}

void
FieldSetMatrixProperty::execute()
{
  std::string matrix1name = guimatrix1name_.get();
  std::string matrix2name = guimatrix2name_.get();
  std::string matrix3name = guimatrix3name_.get();
    
  FieldHandle  handle;
  FieldIPort   *iport;
  FieldOPort   *oport;
  MatrixHandle fhandle;
  MatrixIPort  *ifport;
        
  if(!(iport = static_cast<FieldIPort *>(get_input_port("Field"))))
  {
    error("Could not find 'Field' input port");
    return;
  }
      
  if (!(iport->get(handle)))
  {   
    error("Could not retrieve 'Field' from input port");
  }
   
  if (handle.get_rep() == 0)
  {
    error("No field on input port");
  }
  
  // Scan matrix input port 1
  if (!(ifport = static_cast<MatrixIPort *>(get_input_port("Matrix1"))))
  {
    error("Could not find matrix 1 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->set_property(matrix1name,fhandle,false);
  }

  // Scan matrix input port 2     
  if (!(ifport = static_cast<MatrixIPort *>(get_input_port("Matrix2"))))
  {
    error("Could not find matrix 2 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->set_property(matrix2name,fhandle,false);
  }

  // Scan matrix input port 3     
  if (!(ifport = static_cast<MatrixIPort *>(get_input_port("Matrix3"))))
  {
    error("Could not find matrix 3 input port");
    return;
  }
        
  if (ifport->get(fhandle))
  {
    handle->set_property(matrix3name,fhandle,false);
  }
        
  // Now post the output
        
  if (!(oport = static_cast<FieldOPort *>(get_oport("Field"))))
  {
    error("Could not find field output port");
    return;
  }
  
  handle->generation++;            
  oport->send(handle);
}

void
FieldSetMatrixProperty::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


} // end namespace

