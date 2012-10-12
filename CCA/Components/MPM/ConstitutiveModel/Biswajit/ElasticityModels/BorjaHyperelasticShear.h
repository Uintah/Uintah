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

#ifndef __BORJA_HYPERELASTIC_SHEAR_H__
#define __BORJA_HYPERELASTIC_SHEAR_H__

#include "ShearStressModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class BorjaHyperelasticShear
   *  \brief An isotropic Borja-Tamagnini-Amorosi shear stress model.
   *         Ref: Borja et al., 1997, "Coupling plasticity and energy-conserving
   *              elasticity models for clays", J. Geotech. Geoenv. Engg., 123(10),
   *              p. 948.
   *  \author Biswajit Banerjee 
   *
  */
  class BorjaHyperelasticShear : public ShearStressModel {

  private:

    double d_mu0;    // Reference shear modulus
    double d_p0;     // Reference pressure
    double d_epsv0;  // Reference volumetric strain
    double d_alpha;  // scaling parameter

    BorjaHyperelasticShear& operator=(const BorjaHyperelasticShear &smm);

  public:
         
    /*! Construct a Borja shear stress model. */
    BorjaHyperelasticShear(ProblemSpecP& ps);

    /*! Construct a copy of Borja shear stress model. */
    BorjaHyperelasticShear(const BorjaHyperelasticShear* smm);

    /*! Destructor of Borja shear stress model.   */
    virtual ~BorjaHyperelasticShear();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /*! Compute the shear stress */
    void computeShearStress(const DeformationState* state, Matrix3& stress);
  };
} // End namespace Uintah
      
#endif  // __BORJA_HYPERELASTIC_SHEAR_H__

