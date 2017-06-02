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
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, EROSIONS OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __EROSION_MODEL_H__
#define __EROSION_MODEL_H__

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Patch.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>


namespace Uintah {


  class ErosionModel {
  public:
    ErosionModel( ProblemSpecP& ps,
                  MPMFlags* Mflags,
                  SimulationState* sharedState  );

    ~ErosionModel();

    void outputProblemSpec(ProblemSpecP& ps);


    void addComputesAndRequires(Task             * task,
                                const MPMMaterial* matl);

    void carryForward( const PatchSubset * patches,
                       const MPMMaterial * matl,
                       DataWarehouse     * old_dw,
                       DataWarehouse     * new_dw);

    void addParticleState(std::vector<const VarLabel*>& from,
                          std::vector<const VarLabel*>& to);

    void addInitialComputesAndRequires(Task* task,
                                       const MPMMaterial* matl);

    void initializeLabels(const Patch       * patch,
                          const MPMMaterial * matl,
                          DataWarehouse     * new_dw);


    void updateStress_Erosion( ParticleSubset  * pset,
                               DataWarehouse   * old_dw,
                               DataWarehouse   * new_dw);
                               
    void updateVariables_Erosion( ParticleSubset              * pset,
                                  const ParticleVariable<int>     & pLocalized,
                                  const ParticleVariable<Matrix3> & pFOld,
                                  ParticleVariable<Matrix3>       & pFNew,
                                  ParticleVariable<Matrix3>       & pVelGrad );
                                       
                               
    bool d_doEorsion = false;

    enum  erosionAlgo { ZeroStress,     // set stress tensor to zero
                        AllowNoTension, // retain compressive mean stress after failue
                        AllowNoShear,   // retain mean stress after failure - no deviatoric stress
                        none };

    erosionAlgo d_algo  = erosionAlgo::none;

  private:
    std::string d_algoName = "none";
    
    SimulationState* d_sharedState;
    double           d_charTime;
    MPMLabel*        d_lb;
    
    const VarLabel* pTimeOfLocLabel;
    const VarLabel* pTimeOfLocLabel_preReloc;
  };
} // End namespace Uintah



#endif  // __EROSION_MODEL_H__

