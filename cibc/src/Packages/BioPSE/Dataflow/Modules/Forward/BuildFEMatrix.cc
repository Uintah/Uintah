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

#include <Core/Algorithms/Math/MathAlgo.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>

namespace BioPSE {

using namespace SCIRun;

class BuildFEMatrix : public Module {
public:
  BuildFEMatrix(GuiContext*);

  virtual void execute();

  GuiInt uiUseCond_;
  GuiInt uiUseBasis_;  
  
  int gen_;
  vector<vector<double> > DataBasis_;
  MatrixHandle BasisMatrix_;

  void convert_tensortable(FieldHandle field, MatrixHandle& matrix);  
};


DECLARE_MAKER(BuildFEMatrix)
BuildFEMatrix::BuildFEMatrix(GuiContext* ctx)
  : Module("BuildFEMatrix", ctx, Source, "Forward", "BioPSE"),
    uiUseCond_(ctx->subVar("UseCondTCL")),
    uiUseBasis_(ctx->subVar("UseBasisTCL"))
{
}


void BuildFEMatrix::execute()
{
  FieldHandle Field;
  MatrixHandle Conductivity;
  MatrixHandle SysMatrix;
  
  if (!(get_input_handle("Mesh",Field,true))) return;
  get_input_handle("ConductivityTable",Conductivity,false);
  
  if (inputs_changed_ || !oport_cached("Stiffness Matrix"))
  {
    SCIRunAlgo::MathAlgo numericalgo(this);
  
    if (uiUseBasis_.get())
    {
      if (Conductivity.get_rep()!=0)
      {
        Conductivity.detach();
      }
      else
      {
        convert_tensortable(Field,Conductivity);
      }
      
      if (Conductivity.get_rep())
      {
        int nconds = Conductivity->nrows();
        if ((Field->mesh()->generation != gen_)&&(BasisMatrix_.get_rep()!=0))
        {
        
          MatrixHandle con = scinew DenseMatrix(nconds,1);
          double* data = con->get_data_pointer();
          for (int i=0; i<nconds;i++) data[i] = 0.0;
          if(!(numericalgo.BuildFEMatrix(Field,BasisMatrix_,-1,con))) return;
          int nconds = Conductivity->nrows();
          
          DataBasis_.resize(nconds);
          for (int s=0; s< nconds; s++)
          {
            MatrixHandle temp;
            data[s] = 1.0;
            if(!(numericalgo.BuildFEMatrix(Field,temp,-1,con))) return;
            SparseRowMatrix *m = temp->sparse();
            DataBasis_[s].resize(m->nnz);
            for (int p=0; p< m->nnz; p++)
            {
              DataBasis_[s][p] = m->a[p];
            }
            data[s] = 0.0;
          }

          gen_ = Field->mesh()->generation;
        }

        SysMatrix = BasisMatrix_;
        SysMatrix.detach();
        SparseRowMatrix *m = SysMatrix->sparse();
        double *sum = m->a;
        for (int s=0; s<nconds; s++)
        {
          double weight = Conductivity->get(s,0);
          for (unsigned int p=0; p < DataBasis_[s].size(); p++)
            sum[p] += weight * DataBasis_[s][p];
        }
      }
      
      return;
    }
 
 
    if(!(numericalgo.BuildFEMatrix(Field,SysMatrix,-1,Conductivity))) return;
    
    send_output_handle("Stiffness Matrix",SysMatrix,false);  
  }
}


void 
BuildFEMatrix::convert_tensortable(FieldHandle field, MatrixHandle& matrix)
{
  vector<pair<string,Tensor> > tens;
  
  field->get_property("conductivity_table",tens);
  
  if (tens.size() > 0)
  {
    matrix = scinew DenseMatrix(tens.size(),1);
    double* data = matrix->get_data_pointer();
    for (size_t i; i<tens.size();i++)
    {
      data[i] = tens[i].second.mat_[0][0];
    }
  }
}

} // End namespace SCIRun



