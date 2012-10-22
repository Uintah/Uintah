/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __LINEAR_MELT_TEMP_MODEL_H__
#define __LINEAR_MELT_TEMP_MODEL_H__

#include "MeltingTempModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class LinearMeltTemp
   *  \brief The melting temperature varies linearly with pressure
   *  \author Joseph Peterson, 
   *  \author C-SAFE and Department of Chemistry,
   *  \author University of Utah.
   *
  */
  class LinearMeltTemp : public MeltingTempModel {

  private:
    LinearMeltTemp& operator=(const LinearMeltTemp &mtm);

    bool d_usePressureForm;
    bool d_useVolumeForm;

    double d_Tm0;    // Initial melting temperature (K)
    double d_a;      // Kraut-Kennedy coefficient 
    double d_Gamma;  // Gruneisen gamma 
    double d_b;      // Pressure coefficient (K/Pa)
    double d_K_T;    // Isothermal bulk modulus (Pa)


  public:
         
    /*! Construct a linear melt temp model. */
    LinearMeltTemp();
    LinearMeltTemp(ProblemSpecP& ps);

    /*! Construct a copy of linear melt temp model. */
    LinearMeltTemp(const LinearMeltTemp* mtm);

    /*! Destructor of linear melt temp model.   */
    virtual ~LinearMeltTemp();
         
    virtual void outputProblemSpec(ProblemSpecP& ps);

    /*! Compute the melt temp */
    double computeMeltingTemp(const PlasticityState* state);
  };
} // End namespace Uintah
      
#endif  // __LINEAR_MELT_TEMP_MODEL_H__

