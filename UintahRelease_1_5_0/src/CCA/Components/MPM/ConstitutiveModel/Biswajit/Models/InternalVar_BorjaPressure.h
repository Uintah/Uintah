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

#ifndef __BORJA_PRESSURE_INT_VAR_MODEL_H__
#define __BORJA_PRESSURE_INT_VAR_MODEL_H__


#include "InternalVariableModel.h"    
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace UintahBB {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class InternalVar_BorjaPressure
    \brief The evolution of the consolidation pressure internal variable in the
           Borja model

    Reference:Borja, R.I. and Tamagnini, C.(1998) Cam-Clay plasticity Part III: 
    Extension of the infinitesimal model to include finite strains,
    Computer Methods in Applied Mechanics and Engineering, 155 (1-2),
    pp. 73-95.

    The consolidation presure (p_c) is defined by the rate equation

           1/p_c dp_c/dt = 1/(lambdatilde - kappatilde) depsp_v/dt

           where lambdatilde = material constant
                 kappatilde = material constant
                 epsp_v = volumetric plastic strain

    The incremental update of the consolidation pressure is given by

       (p_c)_{n+1} = (p_c)_n exp[((epse_v)_trial - (epse_v)_{n+1})/(lambdabar - kappabar)]
  */
  ////////////////////////////////////////////////////////////////////////////

  class InternalVar_BorjaPressure : public InternalVariableModel {

  public:

    // Internal variables
    const Uintah::VarLabel* pPcLabel; 
    const Uintah::VarLabel* pPcLabel_preReloc; 

  private:

    // Model parameters
    double d_pc0;
    double d_lambdatilde;
    double d_kappatilde;

         
    // Prevent copying of this class
    // copy constructor
    //InternalVar_BorjaPressure(const InternalVar_BorjaPressure &cm);
    InternalVar_BorjaPressure& operator=(const InternalVar_BorjaPressure &cm);

  public:
    // constructors
    InternalVar_BorjaPressure(Uintah::ProblemSpecP& ps);
    InternalVar_BorjaPressure(const InternalVar_BorjaPressure* cm);
         
    // destructor 
    virtual ~InternalVar_BorjaPressure();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Uintah::Task* task,
                                               const Uintah::MPMMaterial* matl,
                                               const Uintah::PatchSet* patches);

    virtual void addComputesAndRequires(Uintah::Task* task,
                                        const Uintah::MPMMaterial* matl,
                                        const Uintah::PatchSet* patches);

    virtual void allocateCMDataAddRequires(Uintah::Task* task, 
                                           const Uintah::MPMMaterial* matl,
                                           const Uintah::PatchSet* patch, 
                                           Uintah::MPMLabel* lb);

    virtual void allocateCMDataAdd(Uintah::DataWarehouse* new_dw,
                                   Uintah::ParticleSubset* addset,
                                   std::map<const Uintah::VarLabel*, 
                                     Uintah::ParticleVariableBase*>* newState,
                                   Uintah::ParticleSubset* delset,
                                   Uintah::DataWarehouse* old_dw);

    virtual void addParticleState(std::vector<const Uintah::VarLabel*>& from,
                                  std::vector<const Uintah::VarLabel*>& to);

    virtual void initializeInternalVariable(Uintah::ParticleSubset* pset,
                                            Uintah::DataWarehouse* new_dw);

    virtual void getInternalVariable(Uintah::ParticleSubset* pset,
                                     Uintah::DataWarehouse* old_dw,
                                     Uintah::constParticleVariableBase& intvar);

    virtual void allocateAndPutInternalVariable(Uintah::ParticleSubset* pset,
                                                Uintah::DataWarehouse* new_dw,
                                                Uintah::ParticleVariableBase& intvar); 

    virtual void allocateAndPutRigid(Uintah::ParticleSubset* pset, 
                                     Uintah::DataWarehouse* new_dw,
                                     Uintah::constParticleVariableBase& intvar);

    ///////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the internal variable */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeInternalVariable(const ModelState* state) const;

    // Compute derivative of internal variable with respect to volumetric
    // elastic strain
    virtual double computeVolStrainDerivOfInternalVariable(const ModelState* state) const;

 };

} // End namespace Uintah

#endif  // __BORJA_PRESSURE_INT_VAR_MODEL_H__ 
