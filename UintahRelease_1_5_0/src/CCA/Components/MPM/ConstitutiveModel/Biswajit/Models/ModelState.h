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

#ifndef __BB_PLASTICITY_STATE_DATA_H__
#define __BB_PLASTICITY_STATE_DATA_H__

#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PlasticityState.h>
#include <Core/Math/Matrix3.h>

namespace UintahBB {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ModelState
    \brief A structure that store the plasticity state data
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class ModelState: public Uintah::PlasticityState {

  public:
    Uintah::Matrix3 elasticStrain;
    Uintah::Matrix3 elasticStrainTrial;

    double p;         // pressure = tr(sigma)
    double q;         // shear = sqrt(3J2); J2 = 1/2 s:s; s = sigma - p I
    double p_c;       // consolidation pressure

    double epse_v;    // volumetric elastic strain = tr(epse)
    double epse_s;    // deviatoric elastic strain = sqrt(2/3) ||ee||
                      //  ee = epse - 1/3 epse_v I
    double epse_v_tr; // trial volumetric elastic strain
    double epse_s_tr; // trial deviatoric elastic strain 

    double local_var[10]; // Keep aside 10 spaces for local scalar variables

    ModelState();

    ModelState(const ModelState& state);
    ModelState(const ModelState* state);

    ~ModelState();

    ModelState& operator=(const ModelState& state);
    ModelState* operator=(const ModelState* state);
    
  };

} // End namespace Uintah

#endif  // __BB_PLASTICITY_STATE_DATA_H__ 
