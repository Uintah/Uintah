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

#ifndef __HYPOELASTICDEVSTRESS_H__
#define __HYPOELASTICDEVSTRESS_H__


#include "DevStressModel.h"
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {


  class HypoElasticDevStress : public DevStressModel {

  public:
    // constructors
    HypoElasticDevStress();
         
    // destructor 
    virtual ~HypoElasticDevStress();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* ,
                                               const MPMMaterial* );

    virtual void addComputesAndRequires(Task* ,
                                        const MPMMaterial* );

    // For computeStressTensorImplicit
    virtual void addComputesAndRequires(Task* ,
                                        const MPMMaterial* ,
                                        bool );


    virtual void addParticleState(std::vector<const VarLabel*>& ,
                                  std::vector<const VarLabel*>& );

    virtual void initializeInternalVars(ParticleSubset* ,
                                        DataWarehouse* );

    virtual void getInternalVars(ParticleSubset* ,
                                 DataWarehouse* );

    virtual void allocateAndPutInternalVars(ParticleSubset* ,
                                            DataWarehouse* ); 


    virtual void allocateAndPutRigid(ParticleSubset* ,
                                     DataWarehouse* );
                                     
    virtual void computeDeviatoricStressInc( const particleIndex ,
                                             const PlasticityState* ,
                                             DeformationState* ,
                                             const double );

    virtual void updateInternalStresses( const particleIndex,
                                         const Matrix3&,
                                         DeformationState* ,
                                         const double  );

    virtual void rotateInternalStresses( const particleIndex,
                                         const Matrix3&); 
  };

} // End namespace Uintah

#endif  // __HYPOELASTICDEVSTRESS_H__
