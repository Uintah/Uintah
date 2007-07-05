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
#include <Core/Datatypes/String.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/StringPort.h>

#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class MatrixToString : public Module {
public:
  MatrixToString(GuiContext*);
  virtual void execute();
};


DECLARE_MAKER(MatrixToString)
MatrixToString::MatrixToString(GuiContext* ctx)
  : Module("MatrixToString", ctx, Source, "Converter", "ModelCreation")
{
}

void MatrixToString::execute()
{
  MatrixHandle Mat;
  StringHandle Str;
  
  if(!(get_input_handle("Matrix",Mat,true))) return;
  
  std::ostringstream oss;
  
  if (Mat->is_sparse())
  {
    SparseRowMatrix* spr = dynamic_cast<SparseRowMatrix*>(Mat.get_rep());
    int *rr = spr->rows;
    int *cc = spr->columns;
    double *d  = spr->a;
    int m   = spr->nrows();
    int n   = spr->ncols();
    
    oss << "Sparse Matrix ("<<m<<"x"<<n<<"):\n";
    if ((rr)&&(cc)&&(d))
    {
      for (int r = 0; r < m; r++)
      {
        for (int c=rr[r]; c<rr[r+1];c++)
        {
          oss << "["<<r<<","<<cc[c]<<"] = " << d[c] << "\n";
        }
      }
    }
  }
  else
  {
    Mat = Mat->dense();
    int m = Mat->nrows();
    int n = Mat->ncols();
    double* d = Mat->get_data_pointer();
    oss << "Dense Matrix ("<<m<<"x"<<n<<"):\n";
    int k = 0;
    for (int r=0; r<m;r++)
    {
      for (int c=0; c<n;c++)
      {
        oss << d[k++] << " ";
      }
      oss << "\n";
    }
  }
  
  Str = scinew String(oss.str());
  send_output_handle("String",Str,true);
}


} // End namespace ModelCreation


