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

#ifndef __SHEAR_STRESS_MODEL_H__
#define __SHEAR_STRESS_MODEL_H__

#include "DeformationState.h"
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {

  /*! \class ShearStressModel
   *  \brief A generic wrapper for various shear stress models for isotropic
   *         materials.
   *  \author Biswajit Banerjee, 
  */
  class ShearStressModel {

  public:
         
    //! Construct a shear stress model.  
    /*! This is an abstract base class. */
    ShearStressModel();

    //! Destructor of shear stress model.  
    /*! Virtual to ensure correct behavior */
    virtual ~ShearStressModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Compute the shear stress using the time increment, the determinant of the deformation gradient,
    // and the symmetric part of the velocity gradient.
    virtual void computeShearStress(const DeformationState* defState,
                                    Matrix3& shearStress) = 0;
  };
} // End namespace Uintah
      
#endif  // __SHEAR_STRESS_MODEL_H__

