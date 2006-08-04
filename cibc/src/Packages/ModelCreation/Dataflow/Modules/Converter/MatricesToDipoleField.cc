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

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/Field.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Core/Algorithms/Converter/ConverterAlgo.h>

#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class MatricesToDipoleField : public Module {
public:
  MatricesToDipoleField(GuiContext*);

  virtual void execute();
};


DECLARE_MAKER(MatricesToDipoleField)
MatricesToDipoleField::MatricesToDipoleField(GuiContext* ctx)
  : Module("MatricesToDipoleField", ctx, Source, "Converter", "ModelCreation")
{
}

void MatricesToDipoleField::execute()
{
  // Define local handles of data objects:
  MatrixHandle Locations, Strengths;
  FieldHandle DipoleField;
  
  // Get the new input data:
  if (!(get_input_handle("Locations",Locations,true))) return;
  if (!(get_input_handle("Strengths",Strengths,true))) return;

  // Only reexecute if the input changed. SCIRun uses simple scheduling
  // that executes every module downstream even if no data has changed:
  if (inputs_changed_ || !oport_cached("DipoleField"))
  {
    SCIRunAlgo::ConverterAlgo algo(this);
    if(!(algo.MatricesToDipoleField(Locations,Strengths,DipoleField))) return;
  
    // send new output if there is any:
    send_output_handle("DipoleField",DipoleField,false);
  }
}

} // End namespace ModelCreation


