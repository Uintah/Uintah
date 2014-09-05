/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef __PTW_SHEAR_MODEL_H__
#define __PTW_SHEAR_MODEL_H__

#include "ShearModulusModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class PTWShear
   *  \brief The shear modulus model used by Preston,Tonks,Wallace
   *         the PTW plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *  \author Copyright (C) 2004 Container Dynamics Group
   *
  */
  class PTWShear : public ShearModulusModel {

  private:

    double d_mu0;     // Material constant 
    double d_alpha;   // Material constant 
    double d_alphap;  // Material constant 
    double d_slope_mu_p_over_mu0; // Material constant (constant A in SCG model)

    PTWShear& operator=(const PTWShear &smm);

  public:
         
    /*! Construct a constant shear modulus model. */
    PTWShear(ProblemSpecP& ps);

    /*! Construct a copy of constant shear modulus model. */
    PTWShear(const PTWShear* smm);

    /*! Destructor of constant shear modulus model.   */
    virtual ~PTWShear();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /*! Compute the shear modulus */
    double computeShearModulus(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __PTW_SHEAR_MODEL_H__

