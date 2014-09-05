/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef __HypoViscoElasticDEVSTRESS_H__
#define __HypoViscoElasticDEVSTRESS_H__


#include "DevStressModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Containers/StaticArray.h>

#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {


  class HypoViscoElasticDevStress : public DevStressModel {

  private:

    std::vector<const VarLabel*> d_sigmaDevLabel;
    std::vector<const VarLabel*> d_sigmaDevLabel_preReloc;
    
    
    std::vector< constParticleVariable<Matrix3 > * > d_sigmaDev;
    std::vector< ParticleVariable<Matrix3>* > d_sigmaDev_new;
    
    unsigned int d_MaxwellElements;        // number of Maxwell Elements
    std::vector<double>  d_tau_MW;         // tau Maxwell Elements
    std::vector<double>  d_inv_tau_MW;     // 1.0/tau Maxwell Elements  (for speed)
    std::vector<double>  d_mu_MW;          // mu Maxwell
    
  public:
    // constructors
    HypoViscoElasticDevStress(ProblemSpecP& ps);
         
    // destructor 
    virtual ~HypoViscoElasticDevStress();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl);

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl);

    // For computeStressTensorImplicit
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        bool SchedParent);


    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual void initializeInternalVars(ParticleSubset* pset,
                                        DataWarehouse* new_dw);

    virtual void getInternalVars(ParticleSubset* pset,
                                 DataWarehouse* old_dw);

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw); 


    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw);
                                     
    virtual void computeDeviatoricStressInc( const particleIndex idx,
                                             const PlasticityState* plaState,
                                             DeformationState* defState,
                                             const double delT);

    virtual void updateInternalStresses( const particleIndex idx,
                                         const Matrix3&,
                                         DeformationState* defState,
                                         const double delT );

    virtual void rotateInternalStresses( const particleIndex idx,
                                         const Matrix3&);

  };

} // End namespace Uintah

#endif  // __HypoViscoElasticDEVSTRESS_H__
