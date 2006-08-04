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
 *  IndicesToDataArray.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class IndicesToDataArray : public Module {
  public:
    IndicesToDataArray(GuiContext*);

    virtual ~IndicesToDataArray();

    virtual void execute();

    virtual void tcl_command(GuiArgs&, void*);
  
};


DECLARE_MAKER(IndicesToDataArray)
IndicesToDataArray::IndicesToDataArray(GuiContext* ctx)
  : Module("IndicesToDataArray", ctx, Source, "TensorVectorMath", "ModelCreation")
{
}

IndicesToDataArray::~IndicesToDataArray(){
}

void IndicesToDataArray::execute()
{
  MatrixHandle Index, Template, Array;
  MatrixIPort* index_iport;
  MatrixIPort* template_iport;
  MatrixOPort* array_oport;
  
  if (!(index_iport = dynamic_cast<MatrixIPort *>(get_iport("Indices"))))
  {
    error("Could not locate input port 'Indices'");
    return;
  }
  index_iport->get(Index);
  if(Index.get_rep() == 0)
  {
    error("No matrix was found on input port 'Indices'");
    return;
  }
  
  if (!(template_iport = dynamic_cast<MatrixIPort *>(get_iport("Template"))))
  {
    error("Could not locate input port 'Template'");
    return;
  }
  
  template_iport->get(Template);
  if(Template.get_rep() == 0)
  {
    error("No matrix was found on input port 'Template'");
    return;
  }
  
  int numindeces = 0;
  
  DenseMatrix *temp = Template->dense();
  DenseMatrix *indices = Index->dense();
  
  if (temp == 0)
  {
    error("Cannot read matrix on the 'Template' input port");
    return;
  }
  
  if (indices == 0)
  {
    error("Cannot read matrix on the 'Indices' input port");
    return;
  }
  
  int temp_m = temp->nrows(); 
  int temp_n = temp->ncols();
  double* temp_data = temp->get_data_pointer();

  int indices_m = temp->nrows(); 
  int indices_n = temp->ncols();
  double* indices_data = temp->get_data_pointer();
  
  if (temp_data == 0)
  {
    error("No data in the Template matrix");
    return;
  }

  if (indices_data == 0)
  {
    error("No data in the Indices matrix");
    return;
  }
  
  // transpose on paper
  if ((indices_m == 1)&&(indices_n > 1)) { indices_m = indices_n; indices_n = 1;}
  
  if ((indices_m == 0)||(indices_n == 0))
  {
    error("Indices matrix is empty");
    return;  
  }

  if ((temp_m == 0)||(temp_n == 0))
  {
    error("Template matrix is empty");
    return;  
  }

  if (indices_n != 1)
  {
    error("Number of columns of index vector is larger than one");
    return;
  }
    
  Array = dynamic_cast<SCIRun::Matrix *>(scinew DenseMatrix(indices_m,temp_n));
  if (Array.get_rep() == 0)
  {
    error("Could not allocate matrix");
    return;
  }    
    
  double* array_data = Array->get_data_pointer();
  
  if (array_data == 0)
  {
    error("Could not allocate matrix");
    return;
  }  
  
  for (size_t p = 0; p < indices_m; p++)
  {
    int index = static_cast<int>(indices_data[p]);
    if ((index < 0)&&(index >= temp_m))
    {
      for (size_t q = 0; q < temp_n ;q++) array_data[q] = 0.0;
    }
    else
    {
      double* t_data = temp_data + temp_n*index;
      for (size_t q = 0; q < temp_n ;q++) array_data[q] = t_data[q];
    }
  }  
  
  if (array_oport = dynamic_cast<MatrixOPort *>(get_oport(0)))
  {
    array_oport->send(Array);
  }      
}


void
 IndicesToDataArray::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


