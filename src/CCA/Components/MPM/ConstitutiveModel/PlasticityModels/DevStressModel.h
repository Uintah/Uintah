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

#ifndef __DEVSTRESSMODEL_H__
#define __DEVSTRESSMODEL_H__


#include "DeformationState.h"
#include "PlasticityState.h"
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/Matrix3.h>

namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*!
    \class  DevStressModel
    \brief  Abstract Base class for Deviatoric Stress models
    \author Todd Harman
  */
  ///////////////////////////////////////////////////////////////////////////

  class DevStressModel {

  private:

  public:
         
    DevStressModel();
    virtual ~DevStressModel();

    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires( Task* task,
                                                const MPMMaterial* matl) {};

    virtual void addComputesAndRequires( Task* task,
                                         const MPMMaterial* matl ) {};

    virtual void addComputesAndRequires( Task* task,
                                         const MPMMaterial* matl,
                                         bool SchedParent ) {};

    virtual void addParticleState( std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to ){};

    virtual void initializeInternalVars( ParticleSubset* pset,
                                         DataWarehouse* new_dw){};

    virtual void getInternalVars( ParticleSubset* pset,
                                  DataWarehouse* old_dw ){};

    virtual void allocateAndPutInternalVars( ParticleSubset* pset,
                                             DataWarehouse* new_dw ){}; 

    virtual void allocateAndPutRigid( ParticleSubset* pset,
                                      DataWarehouse* new_dw ){};
                               
    //__________________________________
    //  where the work is done
    virtual void computeDeviatoricStressInc( const particleIndex idx,         
                                             const PlasticityState* plaState, 
                                             DeformationState* defState,      
                                             const double delT){};            

    virtual void updateInternalStresses( const particleIndex idx,
                                         const Matrix3&,
                                         DeformationState* defState,
                                         const double delT ){};

    virtual void rotateInternalStresses( const particleIndex idx,
                                         const Matrix3&){};
  };
} // End namespace Uintah
      


#endif  // __DEVSTRESSMODEL_H__

