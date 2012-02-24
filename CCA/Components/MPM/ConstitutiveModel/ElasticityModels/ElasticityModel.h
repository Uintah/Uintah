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


#ifndef __ELASTICITY_MODEL_H__
#define __ELASTICITY_MODEL_H__

#include <Core/Math/Matrix3.h>
#include <vector>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/TangentModulusTensor.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include "ElasticityState.h"


namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  ElasticityModel
    \brief  Abstract Base class for elasticity models (calculate yield stress)
    \author Biswajit Banerjee, 
    \warn   Does not apply all possible strain energy density functions, 
            particularly anisotropic hyperelasticity.  Use with care.
  */
  ///////////////////////////////////////////////////////////////////////////

  class ElasticityModel {

  private:

  public:
         
    ElasticityModel();
    virtual ~ElasticityModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) {};

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) {};

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse,
                                        bool SchedParent) {};

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb){};

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw){};

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to){};

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw){}; 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw){}; 

    virtual void updateElastic(const particleIndex idx){};

    // Compute the continuum elastic tangent modulus 
    virtual void tangentModulus(const Matrix3& stress,
                                const double& delT,
                                const MPMMaterial* matl,
                                const particleIndex idx,
                                TangentModulusTensor& Ce){};

    // Compute the shear modulus. 
    virtual double shearModulus() = 0;

    // Compute the bulk modulus. 
    virtual double bulkModulus() = 0;

  };
} // End namespace Uintah
      


#endif  // __ELASTICITY_MODEL_H__

