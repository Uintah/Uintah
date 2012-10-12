/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __BPS_MELT_TEMP_MODEL_H__
#define __BPS_MELT_TEMP_MODEL_H__

#include "MeltingTempModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class BPSMeltTemp
   *  \brief The melt temp model from Burakovsky, Preston, Siblar in
   *         the PTW plasticity model.
   *  \author Biswajit Banerjee, 
   *  \author C-SAFE and Department of Mechanical Engineering,
   *  \author University of Utah.
   *
  */
  class BPSMeltTemp : public MeltingTempModel {

  private:

    double d_B0;       // Bulk modulus (Guinan-Steinberg, 1974)
    double d_dB_dp0;   // Derivative of bulk modulus (Guinan-Steinberg, 1974)
    double d_G0;       // Shear modulus (Guinan-Steinberg, 1974)
    double d_dG_dp0;   // Derivative of shear modulus (Guinan-Steinberg, 1974)
    double d_kappa;  
    double d_z;        
    double d_b2rhoTm;
    double d_alpha;
    double d_lambda;
    double d_a;        
    double d_factor;        
    double d_kb;       // Boltzmann constant

    BPSMeltTemp& operator=(const BPSMeltTemp &mtm);

  public:
         
    /*! Construct a constant melt temp model. */
    BPSMeltTemp(ProblemSpecP& ps);

    /*! Construct a copy of constant melt temp model. */
    BPSMeltTemp(const BPSMeltTemp* mtm);

    /*! Destructor of constant melt temp model.   */
    virtual ~BPSMeltTemp();
         
    virtual void outputProblemSpec(ProblemSpecP& ps);

    /*! Compute the melt temp */
    double computeMeltingTemp(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __BPS_MELT_TEMP_MODEL_H__

