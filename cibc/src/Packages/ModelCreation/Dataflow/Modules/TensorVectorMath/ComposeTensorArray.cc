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
 *  ComposeTensorArray.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>

namespace ModelCreation {

using namespace SCIRun;

class ComposeTensorArray : public Module {
public:
  ComposeTensorArray(GuiContext*);

  virtual ~ComposeTensorArray();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
  private:
  
  bool  use_9tensor_;
};


DECLARE_MAKER(ComposeTensorArray)
ComposeTensorArray::ComposeTensorArray(GuiContext* ctx)
  : Module("ComposeTensorArray", ctx, Source, "TensorVectorMath", "ModelCreation"),
    use_9tensor_(true)
{
}

ComposeTensorArray::~ComposeTensorArray(){
}

void
 ComposeTensorArray::execute()
{

  MatrixHandle eigvec1, eigvec2, eigval1, eigval2, eigval3, tensor, temp;
  MatrixIPort  *iport;
  MatrixOPort  *oport;

  if (!(oport = dynamic_cast<MatrixOPort *>(get_oport("TensorArray"))))
  {
    error("Could not find TensorArray output port");
    return;
  }

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("EigenVector1"))))
  {
    error("Could not find EigenVector1 input port");
    return;
  }
  iport->get(eigvec1);

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("EigenVector2"))))
  {
    error("Could not find EigenVector2 input port");
    return;
  }
  iport->get(eigvec2);

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("EigenValue1"))))
  {
    error("Could not find EigenValue1 input port");
    return;
  }
  iport->get(eigval1);

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("EigenValue2"))))
  {
    error("Could not find EigenValue2 input port");
    return;
  }
  iport->get(eigval2);

  if (!(iport = dynamic_cast<MatrixIPort *>(get_iport("EigenValue3"))))
  {
    error("Could not find EigenValue3 input port");
    return;
  }
  iport->get(eigval3);
  
  int n = 0;
  int eigvec1n, eigvec2n, eigval1n, eigval2n, eigval3n;
  
  eigvec1n = 0;
  if (eigvec1.get_rep())
  {
    temp = dynamic_cast<Matrix *>(eigvec1->dense());
    eigvec1 = temp;
    if (eigvec1->ncols() != 3)
    {
      error("The EigenVector1 matrix needs to have three columns");
      return;
    }
    
    eigvec1n = eigvec1->nrows();
    if (n == 0)
    {
      n = eigvec1n;
    }
    else
    {
      if ((eigvec1n != 1)&&(eigvec1n != n))
      {
        error("The size of EigenVector1 does not have the same components as the other matrices");
        return;
      }
    }
  }

  eigvec2n = 0;
  if (eigvec2.get_rep())
  {
    temp = dynamic_cast<Matrix *>(eigvec2->dense());
    eigvec2 = temp;
    if (eigvec2->ncols() != 3)
    {
      error("The EigenVector2 matrix needs to have three columns");
      return;
    }
    
    eigvec2n = eigvec2->nrows();
    if (n == 0)
    {
      n = eigvec2n;
    }
    else
    {
      if ((eigvec2n != 1)&&(eigvec2n != n))
      {
        error("The size of EigenVector2 does not have the same components as the other matrices");
        return;
      }
    }
  }
  
  eigval1n = 0;
  if (eigval1.get_rep())
  {
    temp = dynamic_cast<Matrix *>(eigval1->dense());
    eigval1 = temp;
    if (eigval1->ncols() != 1)
    {
      error("The EigenValue1 matrix needs to have only one columns");
      return;
    }
    
    eigval1n = eigval1->nrows();
    if (n == 0)
    {
      n = eigval1n;
    }
    else
    {
      if ((eigval1n != 1)&&(eigval1n != n))
      {
        error("The size of EigenValue1 does not have the same components as the other matrices");
        return;
      }
    }
  }

  eigval2n = 0;
  if (eigval2.get_rep())
  {
    temp = dynamic_cast<Matrix *>(eigval2->dense());
    eigval2 = temp;
    if (eigval2->ncols() != 1)
    {
      error("The EigenValue2 matrix needs to have only one columns");
      return;
    }

    eigval2n = eigval2->nrows();
    if (n == 0)
    {
      n = eigval2n;
    }
    else
    {
      if ((eigval2n != 1)&&(eigval2n != n))
      {
        error("The size of EigenValue1 does not have the same components as the other matrices");
        return;
      }
    }
  }

  eigval3n = 0;
  if (eigval3.get_rep())
  {
    temp = dynamic_cast<Matrix *>(eigval3->dense());
    eigval3 = temp;
    if (eigval3->ncols() != 1)
    {
      error("The EigenValue3 matrix needs to have only one columns");
      return;
    }
    
    eigval3n = eigval3->nrows();
    if (n == 0)
    {
      n = eigval3n;
    }
    else
    {
      if ((eigval3n != 1)&&(eigval3n != n))
      {
        error("The size of EigenVector1 does not have the same components as the other matrices");
        return;
      }
    }
  }

  if (n == 0)
  {
    error("All input matrices are empty");
    return;
  }

  if(!eigval1.get_rep())
  {
    eigval1 = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if (eigval1.get_rep() == 0)
    {
      error("Could not allocate matrix");
      return;
    }
    double* dataptr = eigval1->get_data_pointer();
    dataptr[0] = 0.0;
    eigval1n = 1;
  }

  if(!eigval2.get_rep())
  {
    eigval2 = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if (eigval2.get_rep() == 0)
    {
      error("Could not allocate matrix");
      return;
    }
    double* dataptr = eigval2->get_data_pointer();
    dataptr[0] = 0.0;
    eigval2n = 1;
  }

  if(!eigval3.get_rep())
  {
    eigval3 = dynamic_cast<Matrix *>(scinew DenseMatrix(1,1));
    if (eigval3.get_rep() == 0)
    {
      error("Could not allocate matrix");
      return;
    }
    double* dataptr = eigval3->get_data_pointer();
    dataptr[0] = 0.0;
    eigval3n = 1;    
  }

  if (use_9tensor_)
  {
    tensor = dynamic_cast<Matrix *>(scinew DenseMatrix(n,9));
  }
  else
  {
    tensor = dynamic_cast<Matrix *>(scinew DenseMatrix(n,6));  
  }

  if (tensor.get_rep() == 0)
  {
    error("Could not allocate memory for desination matrix");
    return;
  }
  
  double e11,e12,e13,e21,e22,e23,e31,e32,e33;
  double ev1, ev2, ev3;
  double ea1,ea2,ea3;
  double norm;
  
  double* eigvec1ptr = 0;
  double* eigvec2ptr = 0;
  double* eigval1ptr = 0;
  double* eigval2ptr = 0;
  double* eigval3ptr = 0;
  double* tensptr    = 0;
  if (eigvec1.get_rep()) eigvec1ptr = eigvec1->get_data_pointer();
  if (eigvec2.get_rep()) eigvec2ptr = eigvec2->get_data_pointer();
  if (eigval1.get_rep()) eigval1ptr = eigval1->get_data_pointer();
  if (eigval2.get_rep()) eigval2ptr = eigval2->get_data_pointer();
  if (eigval3.get_rep()) eigval3ptr = eigval3->get_data_pointer();
  if (tensor.get_rep()) tensptr    =  tensor->get_data_pointer();
  
  if (eigvec1n == 0)
  {
    if (use_9tensor_)
    {
      for (int p=0; p<n; p++)
      {
         ev1 = *eigval1ptr;
         tensptr[0] = ev1;
         tensptr[1] = 0.0;
         tensptr[2] = 0.0;
         tensptr[3] = 0.0;
         tensptr[4] = ev1;
         tensptr[5] = 0.0;
         tensptr[6] = 0.0;
         tensptr[7] = 0.0;
         tensptr[8] = ev1;
         if (eigval1n > 1) eigval1ptr++;
         tensptr += 9; 
      }
    }
    else
    {
      for (int p=0; p<n; p++)
      {
         ev1 = *eigval1ptr;
         tensptr[0] = ev1;
         tensptr[1] = 0.0;
         tensptr[2] = 0.0;
         tensptr[3] = ev1;
         tensptr[4] = 0.0;
         tensptr[5] = ev1;
         if (eigval1n > 1) eigval1ptr++;
         tensptr += 6; 
      }    
    }
  }

  if ((eigvec2n == 0)&&(eigvec1n > 0))
  {
    for (int p=0; p<n; p++)
    {
       ev1 = *eigval1ptr;
       ev2 = *eigval2ptr;
       ev3 = *eigval2ptr;
       e11 = eigvec1ptr[0];
       e12 = eigvec1ptr[1];
       e13 = eigvec1ptr[2];
      
       // normalize vector
       norm = sqrt(e11*e11+e12*e12+e13*e13);
       if (norm == 0)  
       {
//        error("One of the EigenVectors is (0,0,0)");
//        return;
          norm = 1.0;
       }
       e11 = e11/norm; e12 = e12/norm; e13 = e13/norm;
             
       // Get absoulte values of the eigen vector
       ea1 = fabs(e11); ea2 = fabs(e12); ea3 = fabs(e13);

       // determine the two other normal directions
       if ((ea1 >= ea2)&&(ea1 >= ea3))
       {
         e21 = e12+e13; e22 = -e11; e23 = -e11;
       }
       else if ((ea2 >= ea1)&&(ea2 >= ea3))
       {
         e21 = -e12; e22 = e11+e13; e23 = -e12;       
       }
       else
       {
         e21 = -e13; e22 = -e13; e23 = e11+e12;              
       }

       // normalize vector
       norm = sqrt(e21*e21+e22*e22+e23*e23);
       e21 = e21/norm; e22 = e22/norm; e23 = e23/norm;
       
       // cross product
       e31 = e12*e23 - e13*e22;
       e32 = e13*e21 - e11*e23;
       e33 = e11*e22 - e12*e21;

       if (use_9tensor_)
       {       
          tensptr[0] = (ev1*e11*e11+ev2*e21*e21+ev3*e31*e31);
          tensptr[1] = (ev1*e12*e11+ev2*e22*e21+ev3*e32*e31);
          tensptr[2] = (ev1*e13*e11+ev2*e23*e21+ev3*e33*e31);
          tensptr[3] = tensptr[1];
          tensptr[4] = (ev1*e12*e12+ev2*e22*e22+ev3*e32*e32);
          tensptr[5] = (ev1*e12*e13+ev2*e22*e23+ev3*e32*e33);
          tensptr[6] = tensptr[2];
          tensptr[7] = tensptr[5];
          tensptr[8] = (ev1*e13*e13+ev2*e23*e23+ev3*e33*e33);
          tensptr += 9; 
       }
       else
       {
          tensptr[0] = (ev1*e11*e11+ev2*e21*e21+ev3*e31*e31);
          tensptr[1] = (ev1*e12*e11+ev2*e22*e21+ev3*e32*e31);
          tensptr[2] = (ev1*e13*e11+ev2*e23*e21+ev3*e33*e31);
          tensptr[3] = (ev1*e12*e12+ev2*e22*e22+ev3*e32*e32);
          tensptr[4] = (ev1*e12*e13+ev2*e22*e23+ev3*e32*e33);
          tensptr[5] = (ev1*e13*e13+ev2*e23*e23+ev3*e33*e33);       
          tensptr += 6; 
       }    
       if (eigvec1n > 1) eigvec1ptr+=3;
       if (eigval1n > 1) eigval1ptr++;
       if (eigval2n > 1) eigval2ptr++;

    }
  }

  if ((eigvec2n > 0)&&(eigvec1n > 0))
  {
    for (int p=0; p<n; p++)
    {
       ev1 = *eigval1ptr;
       ev2 = *eigval2ptr;
       ev3 = *eigval3ptr;
       e11 = eigvec1ptr[0];
       e12 = eigvec1ptr[1];
       e13 = eigvec1ptr[2];

       e21 = eigvec2ptr[0];
       e22 = eigvec2ptr[1];
       e23 = eigvec2ptr[2];
      
       // normalize vector
       norm = sqrt(e11*e11+e12*e12+e13*e13);
       if (norm == 0)
       {
        error("One of the eigenvectors is (0,0,0)");
       }
       e11 = e11/norm; e12 = e12/norm; e13 = e13/norm;

       // normalize vector
       norm = sqrt(e21*e21+e22*e22+e23*e23);
       if (norm == 0)
       {
        error("One of the eigenvectors is (0,0,0)");
       }
       e21 = e21/norm; e22 = e22/norm; e23 = e23/norm;
       
       // cross product
       e31 = e12*e23 - e13*e22;
       e32 = e13*e21 - e11*e23;
       e33 = e11*e22 - e12*e21;
       
       if (use_9tensor_)
       {       
          tensptr[0] = (ev1*e11*e11+ev2*e21*e21+ev3*e31*e31);
          tensptr[1] = (ev1*e12*e11+ev2*e22*e21+ev3*e32*e31);
          tensptr[2] = (ev1*e13*e11+ev2*e23*e21+ev3*e33*e31);
          tensptr[3] = tensptr[1];
          tensptr[4] = (ev1*e12*e12+ev2*e22*e22+ev3*e32*e32);
          tensptr[5] = (ev1*e12*e13+ev2*e22*e23+ev3*e32*e33);
          tensptr[6] = tensptr[2];
          tensptr[7] = tensptr[5];
          tensptr[8] = (ev1*e13*e13+ev2*e23*e23+ev3*e33*e33);
          tensptr += 9; 
       }
       else
       {
          tensptr[0] = (ev1*e11*e11+ev2*e21*e21+ev3*e31*e31);
          tensptr[1] = (ev1*e12*e11+ev2*e22*e21+ev3*e32*e31);
          tensptr[2] = (ev1*e13*e11+ev2*e23*e21+ev3*e33*e31);
          tensptr[3] = (ev1*e12*e12+ev2*e22*e22+ev3*e32*e32);
          tensptr[4] = (ev1*e12*e13+ev2*e22*e23+ev3*e32*e33);
          tensptr[5] = (ev1*e13*e13+ev2*e23*e23+ev3*e33*e33);       
          tensptr += 6; 
       }  

       if (eigvec1n > 1) eigvec1ptr+=3;
       if (eigvec2n > 1) eigvec2ptr+=3;
       if (eigval1n > 1) eigval1ptr++;
       if (eigval2n > 1) eigval2ptr++;
       if (eigval3n > 1) eigval3ptr++;
    }
  }

  oport->send(tensor);
}

void
 ComposeTensorArray::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


