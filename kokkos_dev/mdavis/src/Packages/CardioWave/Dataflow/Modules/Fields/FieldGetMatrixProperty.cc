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
 *  FieldGetMatrixPropertyProperty.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdString.h>

using namespace SCIRun;
using namespace std;

class FieldGetMatrixProperty : public Module {
public:
  FieldGetMatrixProperty(GuiContext*);

  virtual ~FieldGetMatrixProperty();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
  
private:
  GuiString             guimatrix1name_;
  GuiString             guimatrix2name_;
  GuiString             guimatrix3name_;
  GuiString             guimatrixs_;
};


DECLARE_MAKER(FieldGetMatrixProperty)
  FieldGetMatrixProperty::FieldGetMatrixProperty(GuiContext* ctx)
    : Module("FieldGetMatrixProperty", ctx, Source, "Fields", "CardioWave"),
      guimatrix1name_(ctx->subVar("matrix1-name")),
      guimatrix2name_(ctx->subVar("matrix2-name")),
      guimatrix3name_(ctx->subVar("matrix3-name")),
      guimatrixs_(ctx->subVar("matrix-selection"))
{

}

FieldGetMatrixProperty::~FieldGetMatrixProperty(){
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
  MatrixOPort *ofport;
  MatrixHandle fhandle;
        
  if(!(iport = static_cast<FieldIPort *>(getIPort("field"))))
    {
      error("Could not find field input port");
      return;
    }

  if (!(iport->get(handle)))
    {   
      warning("No field connected to the input port");
      return;
    }

  if (handle.get_rep() == 0)
    {   
      warning("Empty field connected to the input port");
      return;
    }

  size_t nprop = handle->nproperties();

  for (size_t p=0;p<nprop;p++)
  {
    handle->get_property(handle->get_property_name(p),fhandle);
    if (fhandle.get_rep()) matrixlist += "{" + handle->get_property_name(p) + "} ";
  }

  guimatrixs_.set(matrixlist);
  ctx->reset();
 
 
   if (!(ofport = static_cast<MatrixOPort *>(get_oport("matrix1"))))
    {
      error("Could not find matrix 1 output port");
      return; 
    }
 
  NrrdIPort *niport = static_cast<NrrdIPort *>(getIPort("name1"));
  if (niport)
    {
      NrrdDataHandle matrixH;
      niport->get(matrixH);
      if (matrixH.get_rep() != 0)
        {
          NrrdString matrixstring(matrixH); 
          matrix2name = matrixstring.getstring();
          guimatrix1name_.set(matrix1name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(matrix1name))
    {
      handle->get_property(matrix1name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }

 
  if (!(ofport = static_cast<MatrixOPort *>(get_oport("matrix2"))))
    {
      error("Could not find matrix 2 output port");
      return; 
    }
 
   niport = static_cast<NrrdIPort *>(getIPort("name2"));
  if (niport)
    {
      NrrdDataHandle matrixH;
      niport->get(matrixH);
      if (matrixH.get_rep() != 0)
        {
          NrrdString matrixstring(matrixH); 
          matrix2name = matrixstring.getstring();
          guimatrix2name_.set(matrix2name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(matrix2name))
    {
      handle->get_property(matrix2name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }
        
 
  if (!(ofport = static_cast<MatrixOPort *>(get_oport("matrix3"))))
    {
      error("Could not find matrix 3 output port");
      return; 
    }
 
  niport = static_cast<NrrdIPort *>(getIPort("name3"));
  if (niport)
    {
      NrrdDataHandle matrixH;
      niport->get(matrixH);
      if (matrixH.get_rep() != 0)
        {
          NrrdString matrixstring(matrixH); 
          matrix3name = matrixstring.getstring();
          guimatrix3name_.set(matrix3name);
          ctx->reset();
        }
    } 
 
  if (handle->is_property(matrix3name))
    {
      handle->get_property(matrix3name,fhandle);
      if (handle.get_rep()) ofport->send(fhandle);
    }

        
}

void
FieldGetMatrixProperty::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}




