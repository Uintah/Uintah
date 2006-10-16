/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <string>
#include <float.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class DataArrayMeasure : public Module {
public:
  DataArrayMeasure(GuiContext*);
  virtual void execute();

private:
  GuiString guimeasure_;
};


DECLARE_MAKER(DataArrayMeasure)
DataArrayMeasure::DataArrayMeasure(GuiContext* ctx)
  : Module("DataArrayMeasure", ctx, Source, "TensorVectorMath", "ModelCreation"),
  guimeasure_(ctx->subVar("measure"))
{
}


void
DataArrayMeasure::execute()
{
  MatrixHandle ArrayData;
  MatrixHandle Measure;
  
  if (!(get_input_handle("Array",ArrayData,true))) return;
  
  if (inputs_changed_ || guimeasure_.changed() ||
      !oport_cached("Measure"))
  {
    DenseMatrix* mat = ArrayData->dense();
    
    int m = mat->nrows();
    int n = mat->ncols();
    double* data = mat->get_data_pointer();
    
    std::string method = guimeasure_.get();
    
    Measure = scinew DenseMatrix(1, n);
    double* dest = Measure->get_data_pointer();
    
    if (method == "Sum")
    {
      for (int p=0; p<n; p++) dest[p] = 0.0;
      for (int q=0; q<m; q++) for (int p=0; p<n; p++) dest[p] += data[q*n+p]; 
    }
    else if (method == "Mean")
    {
      for (int p=0; p<n; p++) dest[p] = 0.0;
      for (int q=0; q<m; q++) for (int p=0; p<n; p++) dest[p] += data[q*n+p]; 
      for (int p=0; p<n; p++) dest[p] /= m;  
    }
    else if (method == "Variance")
    {
      double mean[9];
      for (int p=0; p<n; p++) dest[p] = 0.0;
      for (int q=0; q<m; q++) for (int p=0; p<n; p++) mean[p] += data[q*n+p]; 
      for (int p=0; p<n; p++) mean[p] /= m;  
      for (int q=0; q<m; q++) for (int p=0; p<n; p++) dest[p] += (data[q*n+p]-mean[p])*(data[q*n+p]-mean[p]); 
    }
    else if (method == "StdDev")
    {
      if (m == 1)
      {
        for (int p=0; p<n; p++) dest[p] = 0.0;    
      }
      else
      {
        double mean[9];
        for (int p=0; p<n; p++) dest[p] = 0.0;
        for (int q=0; q<m; q++) for (int p=0; p<n; p++) mean[p] += data[q*n+p]; 
        for (int p=0; p<n; p++) mean[p] /= m;  
        for (int q=0; q<m; q++) for (int p=0; p<n; p++) dest[p] += (data[q*n+p]-mean[p])*(data[q*n+p]-mean[p]); 
        for (int p=0; p<n; p++) dest[p] = sqrt(dest[p])/(m-1);      
      }
    }
    else if (method == "Norm")
    {
      for (int p=0; p<n; p++) dest[p] = 0.0;
      for (int q=0; q<m; q++) for (int p=0; p<n; p++) dest[p] += (data[q*n+p])*(data[q*n+p]);
      for (int p=0; p<n; p++) dest[p] = sqrt(dest[p]);       
    }
    else
    {
      if (n > 1)
      {
        error("DataArrayMeasure: This method has not yet been implemented for a vector or tensor");
        return;
      }
      if (method == "Maximum")
      {
        dest[0] = DBL_MIN;
        for (int q=0; q<m; q++) if (data[q] > dest[0]) dest[0] = data[q]; 
      }
      else if (method == "Minimum")
      {
        dest[0] = DBL_MAX;
        for (int q=0; q<m; q++) if (data[q] < dest[0]) dest[0] = data[q];     
      }
      else if (method == "Median")
      {
        std::vector<double> v(m);
        for (int q=0; q<m; q++) v[q] = data[q];
        std::sort(v.begin(),v.end());
        if ((m/2)*2 == m)
        {
          dest[0] = 0.5*(v[m/2]+v[(m/2) -1]);
        }
        else
        {
          dest[0] = v[m/2];
        }
      }
      else
      {
        error("DataArrayMeasure: This method has not yet been implemented");
        return;    
      }
    }

    send_output_handle("Measure",Measure,true);
  }
}

} // End namespace ModelCreation


