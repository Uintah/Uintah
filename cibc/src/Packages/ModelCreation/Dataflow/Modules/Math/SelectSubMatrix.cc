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
#include <Core/Algorithms/Converter/ConverterAlgo.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

namespace ModelCreation {

using namespace SCIRun;

class SelectSubMatrix : public Module {
public:
  SelectSubMatrix(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(SelectSubMatrix)
SelectSubMatrix::SelectSubMatrix(GuiContext* ctx)
  : Module("SelectSubMatrix", ctx, Source, "Math", "ModelCreation")
{
}


void
SelectSubMatrix::execute()
{
  MatrixHandle matrix, rows, columns;
  
  if (!(get_input_handle("Matrix",matrix,true))) return;
  get_input_handle("Rows",rows,false);
  get_input_handle("Columns",columns,false);
  
  if ( (rows.get_rep() == 0)&&(columns.get_rep()))
  {
    error("No row or column indices are specified");
    return;
  }

  if (inputs_changed_ || !oport_cached("Matrix"))
  {
    MatrixHandle output;
  
    SCIRunAlgo::MathAlgo malgo(this);
    SCIRunAlgo::ConverterAlgo calgo(this);
    
    std::vector<unsigned int> ri, ci;
    
    if (rows.get_rep() && columns.get_rep())
    {
      if (!(calgo.MatrixToUnsignedIntVector(rows,ri))) return;
      if (!(calgo.MatrixToUnsignedIntVector(columns,ci))) return;
      if (!(malgo.MatrixSelectSubMatrix(matrix,output,ri,ci))) return;
    }
    else if (rows.get_rep() == 0)
    {
      if (!(calgo.MatrixToUnsignedIntVector(columns,ci))) return;
      if (!(malgo.MatrixSelectColumns(matrix,output,ci))) return;    
    }
    else
    {
      if (!(calgo.MatrixToUnsignedIntVector(rows,ri))) return;
      if (!(malgo.MatrixSelectRows(matrix,output,ri))) return;        
    }
  
    send_output_handle("Matrix", output);
  }
}

} // End namespace ModelCreation


