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
 *  IndicesToTensors: Change a Field of indices (ints) into a Field or Tensors,
 *                      where the Tensor values are looked up in the
 *                      conductivity_table for each index
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>

namespace BioPSE {

using namespace SCIRun;

class GetActivationTimes : public Module {
public:
  GetActivationTimes(GuiContext *context);
  virtual ~GetActivationTimes();
  virtual void execute();
};


DECLARE_MAKER(GetActivationTimes)


GetActivationTimes::GetActivationTimes(GuiContext *context)
  : Module("GetActivationTimes", context, Filter, "Forward", "BioPSE")
{
}

GetActivationTimes::~GetActivationTimes()
{
}

void GetActivationTimes::execute()
{
  const double sample_rate = 1.0;

   // Get input matrix of potentials
   MatrixHandle potentials;
   if (!get_input_handle("Potentials", potentials)) return;

  // Uncomment this section to create multiple activation times
  
   // create vector to store activation times
//   vector<double *> times;
  
  
//   while (true)
//   {
//     double *time = new double[potentials->nrows()];

     // compute the time.

//    times.push_back(time);
//    break;
//  }

  MatrixHandle output = scinew DenseMatrix(potentials->nrows(), 1);
  
  double diff;         // current negative slope
  double max_diff;     // maximum negative slope
  int k;               // index of maximum negative slope
 
  // for each point, find the maximum negative slope 
  for (int i = 0; i < potentials->nrows(); i++)
  {
    // initialiize the maximum slope for first set of points
    max_diff = potentials->get(i,1) - potentials->get(i,0);
    k = 1;

    for (int j = 2; j < potentials->ncols(); j++)
    {
      diff = potentials->get(i,j) - potentials->get(i,j-1);
      
      // if current point is maximum negative slope, set values
      if(diff < max_diff)
      {
        max_diff = diff;
        k = j;
      }
    }
    // send time to the output 
    output->put(i,0, k/sample_rate);
  }

  send_output_handle("ActivationTimes", output);
}


} // End namespace BioPSE
