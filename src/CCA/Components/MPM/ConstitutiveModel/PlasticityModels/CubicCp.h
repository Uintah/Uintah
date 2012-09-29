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

#ifndef __CUBIC_SPECIFIC_HEAT_MODEL_H__
#define __CUBIC_SPECIFIC_HEAT_MODEL_H__

#include "SpecificHeatModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /*! \class CubicCp
   *  \brief The specfic heat varies from the Debye cubic temperature dependence 
   *    to constant at the asymptotic upper limit is reached.
   *  \author Joseph Peterson, 
   *  \author Department of Chemistry,
   *  \author University of Utah.
   *
  */
  class CubicCp : public SpecificHeatModel {

  private:
    CubicCp& operator=(const CubicCp &smm);

    double d_a;
    double d_b;
    double d_beta; // Volumetric coefficient of thermal expansion, K^-1
    double d_c0;   // kgK/J
    double d_c1;   // kgK/J
    double d_c2;   // kgK/J
    double d_c3;   // kgK/J
      
    /*! A helper function to compute the Debye Temperature */
    double computeDebyeT(double, double);
      
  public:
         
    /*! Construct a constant specfic heat model. */
    CubicCp();
    CubicCp(ProblemSpecP& ps);

    /*! Construct a copy of constant specfic heat model. */
    CubicCp(const CubicCp* smm);

    /*! Destructor of constant specfic heat model.   */
    virtual ~CubicCp();
         
    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /*! Compute the specfic heat */
    double computeSpecificHeat(const PlasticityState* state);
      
    /*! A helper function to compute the Gruneisen coefficient */
    double computeGamma(double,double);
      
  };
} // End namespace Uintah
      
#endif  // __CUBIC_SPECIFIC_HEAT_MODEL_H__

