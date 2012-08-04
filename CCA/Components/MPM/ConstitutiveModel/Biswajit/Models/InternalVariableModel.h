/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __INTERNAL_VARIABLE_MODEL_H__
#define __INTERNAL_VARIABLE_MODEL_H__

#include "ModelState.h"
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Math/Matrix3.h>
#include <vector>
#include <map>


namespace UintahBB {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  InternalVariableModel
    \brief  Abstract base class for the evolution of internal variables
            for plasticity models
  */
  ///////////////////////////////////////////////////////////////////////////

  class InternalVariableModel {

  private:

  public:
         
    InternalVariableModel();
    virtual ~InternalVariableModel();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps) = 0;
         
    // Initial computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Uintah::Task* task,
                                               const Uintah::MPMMaterial* matl,
                                               const Uintah::PatchSet* patches) {};

    // Computes and requires for internal evolution variables
    virtual void addComputesAndRequires(Uintah::Task* task,
                                        const Uintah::MPMMaterial* matl,
                                        const Uintah::PatchSet* patches) {};

    virtual void allocateCMDataAddRequires(Uintah::Task* task, 
                                           const Uintah::MPMMaterial* matl,
                                           const Uintah::PatchSet* patch, 
                                           Uintah::MPMLabel* lb){};

    virtual void allocateCMDataAdd(Uintah::DataWarehouse* new_dw,
                                   Uintah::ParticleSubset* addset,
                                   std::map<const Uintah::VarLabel*, 
                                     Uintah::ParticleVariableBase*>* newState,
                                   Uintah::ParticleSubset* delset,
                                   Uintah::DataWarehouse* old_dw){};

    virtual void addParticleState(std::vector<const Uintah::VarLabel*>& from,
                                  std::vector<const Uintah::VarLabel*>& to){};

    virtual void initializeInternalVariable(Uintah::ParticleSubset* pset,
                                            Uintah::DataWarehouse* new_dw){};

    virtual void getInternalVariable(Uintah::ParticleSubset* pset,
                                     Uintah::DataWarehouse* old_dw){};

    virtual void allocateAndPutInternalVariable(Uintah::ParticleSubset* pset,
                                                Uintah::DataWarehouse* new_dw){}; 

    virtual void allocateAndPutRigid(Uintah::ParticleSubset* pset,
                                     Uintah::DataWarehouse* new_dw){}; 

    ///////////////////////////////////////////////////////////////////////////
    // Get the internal variable
    virtual double getInternalVariable(const Uintah::particleIndex idx) const = 0; 

    ///////////////////////////////////////////////////////////////////////////
    // Update the internal variable
    virtual void updateInternalVariable(const Uintah::particleIndex idx,
                                        const double& value) = 0; 

    ///////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the internal variable and return new value  */
    virtual double computeInternalVariable(const ModelState* state,
                                           const double& delT,
                                           const Uintah::MPMMaterial* matl,
                                           const Uintah::particleIndex idx) = 0;

    ///////////////////////////////////////////////////////////////////////////
    // Compute derivative of internal variable with respect to volumetric
    // elastic strain
    virtual double computeVolStrainDerivOfInternalVariable(const ModelState* state) const = 0;

  };
} // End namespace Uintah
      


#endif  // __INTERNAL_VARIABLE_MODEL_H__

