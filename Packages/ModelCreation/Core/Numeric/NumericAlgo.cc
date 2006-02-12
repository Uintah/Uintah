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

#include <Packages/ModelCreation/Core/Numeric/NumericAlgo.h>
#include <Packages/ModelCreation/Core/Numeric/BuildFEMatrix.h>

#include <Core/Algorithms/Fields/FieldCount.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <Dataflow/Modules/Fields/ApplyMappingMatrix.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

NumericAlgo::NumericAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool NumericAlgo::BuildFEMMatrix(FieldHandle field, MatrixHandle& matrix, int num_proc)
{
  BuildFEMatrixAlgo algo;
  algo.BuildFEMatrix(pr_,field,matrix,num_proc);
}

bool NumericAlgo::ResizeMatrix(MatrixHandle input, MatrixHandle& output, int m, int n)
{ 
  if (input->is_sparse())
  {
    double* val = input->get_val();
    int* row = input->get_row();
    int* col = input->get_col();
    int sm = input->nrows();
    int sn = input->ncols();
    int nnz = input->get_data_size();
  
    int newnnz =  0;
    for (int p=0; p<nnz; p++) if (col[p] < m) newnnz++;
    
    double* newval = scinew double[newnnz];  
    int* newcol = scinew int[newnnz];
    int* newrow = scinew int[n+1];
    
    if ((newval == 0)||(newcol == 0)||(newrow == 0))
    {
      error("ResizeMatrix: Could not allocate output matrix");
      if (newval) delete newval;
      if (newcol) delete newcol;
      if (newrow) delete newrow;
      return (false);
    }
    
    int k = 0;
    for (int p=0; p<nnz; p++) if (col[p] < m) 
    {
      newval[k] = val[k];
      newcol[k] = col[k];
      k++;
    }
    
    int r=0;
    newrow[0] = 0;
    for (int p=1; p<n+1; p++)
    {
      for (int q = row[p-1]; q < row[p]; q++) if (col[q] < m) r++;
      newrow[p] = r;
    }
    
    output = dynamic_cast<Matrix *>(scinew SparseRowMatrix(m,n,newrow,newcol,newnnz,newval));
    if (output.get_rep() == 0)
    {
      error("ResizeMatrix: Could not allocate output matrix");
      if (newval) delete newval;
      if (newcol) delete newcol;
      if (newrow) delete newrow;
      return (false);    
    }
    return (true);
  }
  else
  {
    MatrixHandle mat = dynamic_cast<Matrix *>(input->dense());
    output = dynamic_cast<Matrix *>(scinew DenseMatrix(m,n));
    if (output.get_rep() == 0)
    {
      error("ResizeMatrix: Could not allocate output matrix");
      return (false);
    }
  
    int sm = input->nrows();
    int sn = input->ncols();
    
    double *src = input->get_data_pointer();
    double *dst = output->get_data_pointer();
    
    int p,q;
    for (p=0;(p<m)&&(p<sm);p++)
    {
      for (q=0;(q<n)&&(q<sn);q++)
      {
        dst[q+p*m] = src[q+p*sm];
      }
      for (;q<n;q++) dst[q+p*m] = 0.0;
    }
    for (;p<m;p++)
      for (q=0;q<n;q++) dst[q+p*m]= 0.0;
    return (true);
  }
  
  return (false);
}

} // ModelCreation
