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
 *  DistanceToBoundary.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>

#include <Packages/ModelCreation/Core/Datatypes/SelectionMask.h>
#include <Packages/ModelCreation/Core/Fields/FieldsMath.h>

namespace ModelCreation {

using namespace SCIRun;

class DistanceToBoundary : public Module {
public:
  DistanceToBoundary(GuiContext*);

  virtual ~DistanceToBoundary();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  int fGeneration_;
};


DECLARE_MAKER(DistanceToBoundary)
DistanceToBoundary::DistanceToBoundary(GuiContext* ctx)
  : Module("DistanceToBoundary", ctx, Source, "FieldsData", "ModelCreation"),
  fGeneration_(-1)
{
}

DistanceToBoundary::~DistanceToBoundary(){
}

void DistanceToBoundary::execute()
{
  FieldIPort *field_iport;
  
  if (!(field_iport = dynamic_cast<FieldIPort *>(getIPort(0))))
  {
    error("Could not find Field input port");
    return;
  }
  
  FieldHandle input;
  FieldHandle object;
  
  field_iport->get(input);
  
  if (input.get_rep() == 0)
  {
    warning("No Field was found on input port");
    return;
  }

  FieldsMath fieldmath(this);

  bool update = false;

  if ( fGeneration_ != input->generation ) {
    fGeneration_ = input->generation;
    update = true;
  }

  if (update)
  {
    {
      MatrixHandle dummy;
      if (!(fieldmath.FieldBoundary(input,object,dummy)))
      {
        error("FieldBoundary failed to compute boundary");
        return;
      }
    }

    if (object.get_rep() == 0)
    {
      warning("Could not create boundary field");
      return;
    }

    FieldHandle output;

    if(!(fieldmath.DistanceToField(input,output,object)))
    {
      return;
    }
  
    FieldOPort* output_oport = dynamic_cast<FieldOPort *>(getOPort(0));
    if (output_oport) output_oport->send(output);

    MatrixOPort* data_oport = dynamic_cast<MatrixOPort *>(getOPort(1));
    if (data_oport) 
    {
      MatrixHandle data;
      if(fieldmath.GetFieldData(output,data))
      {
        data_oport->send(data);
      }
      else
      {
        error("Could not retrieve data from field, so no data matrix is generated");
        return;        
      }
    }
  }
}

void
 DistanceToBoundary::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


