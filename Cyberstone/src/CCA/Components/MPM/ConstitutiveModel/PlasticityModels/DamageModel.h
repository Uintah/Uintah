/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef __DAMAGE_MODEL_H__
#define __DAMAGE_MODEL_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Patch.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>


namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class DamageModel
    \brief Abstract base class for damage models
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class DamageModel {
  public:


    enum struct DamageAlgo { brittle,
                             threshold,
                             hancock_mackenzie,
                             johnson_cook,
                             none };

    DamageAlgo Algorithm = DamageAlgo::none;

    DamageModel();
    virtual ~DamageModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    virtual
    void addComputesAndRequires(Task* task,
                                const MPMMaterial* matl);

    virtual
    void carryForward( const PatchSubset* patches,
                       const MPMMaterial* matl,
                       DataWarehouse*     old_dw,
                       DataWarehouse*     new_dw);

    virtual
    void addParticleState(std::vector<const VarLabel*>& from,
                          std::vector<const VarLabel*>& to);

    virtual
    void addInitialComputesAndRequires(Task* task,
                                       const MPMMaterial* matl);

    virtual
    void initializeLabels(const Patch*       patch,
                          const MPMMaterial* matl,
                          DataWarehouse*     new_dw);

    virtual
    void computeSomething( ParticleSubset    * pset,
                           const MPMMaterial * matl,
                           const Patch       * patch,
                           DataWarehouse     * old_dw,
                           DataWarehouse     * new_dw );

    protected:
    MPMLabel* d_lb;
  };
} // End namespace Uintah



#endif  // __DAMAGE_MODEL_H__

