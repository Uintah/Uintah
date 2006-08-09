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
#include <Core/Algorithms/Math/MathAlgo.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#
namespace ModelCreation {

using namespace SCIRun;

class AppendMatrix : public Module {
public:
  AppendMatrix(GuiContext*);
  virtual void execute();
  
private:
  GuiString guiroc_;
};


DECLARE_MAKER(AppendMatrix)
AppendMatrix::AppendMatrix(GuiContext* ctx)
  : Module("AppendMatrix", ctx, Source, "Math", "ModelCreation"),
    guiroc_(ctx->subVar("row-or-column"))
{
}

void AppendMatrix::execute()
{
  MatrixHandle base;
  std::vector<MatrixHandle> matrices;
  
  get_input_handle("BaseMatrix",base,false);
  get_dynamic_input_handles("AppendMatrix",matrices,false);
  
  if (inputs_changed_ || guiroc_.changed() || !oport_cached("Matrix"))
  {
    std::string roc = guiroc_.get();
    SCIRunAlgo::MathAlgo malgo(this);
    MatrixHandle matrix;
    
    if (roc == "column")
    {
      std::vector<unsigned int> dummy;
      matrix = base;
      for (int p=0; p<static_cast<int>(matrices.size());p++)
      { 
        if (!(malgo.MatrixAppendColumns(matrix,matrix,matrices[p],dummy))) return;
      }
    }
    else
    {
      std::vector<unsigned int> dummy;
      matrix = base;
      for (int p=0; p<static_cast<int>(matrices.size());p++)
      { 
        if (!(malgo.MatrixAppendRows(matrix,matrix,matrices[p],dummy))) return;
      }    
    }
  
    send_output_handle("Matrix",matrix,false);
  }
}

} // End namespace ModelCreation


