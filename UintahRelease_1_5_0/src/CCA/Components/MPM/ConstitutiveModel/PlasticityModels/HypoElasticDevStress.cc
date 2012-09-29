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
#include "HypoElasticDevStress.h"
#include <Core/Math/FastMatrix.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <cmath>
using namespace std;
using namespace Uintah;

// constructor
HypoElasticDevStress::HypoElasticDevStress(){}

// destructor
HypoElasticDevStress::~HypoElasticDevStress(){}


//______________________________________________________________________
//
void HypoElasticDevStress::computeDeviatoricStressInc( const particleIndex ,
                                                       const PlasticityState* plaState ,
                                                       DeformationState* defState ,
                                                       const double delT ){ 
  //proc0cout << " HypoElasticDevStress:computeDevStessInc " << endl;
  double mu = plaState->shearModulus;
  defState->devStressInc = defState->tensorEta * (2.0 * mu * delT);
}


//______________________________________________________________________
//    EMPTY METHODS & FUNCTIONS
//______________________________________________________________________

void HypoElasticDevStress::outputProblemSpec(ProblemSpecP& ps){}

void 
HypoElasticDevStress::addInitialComputesAndRequires(Task* ,
                                                    const MPMMaterial* ){}

void 
HypoElasticDevStress::addComputesAndRequires(Task* ,
                                             const MPMMaterial* ){}

void
HypoElasticDevStress::addComputesAndRequires(Task* ,
                                             const MPMMaterial*,
                                             bool ) {}

void 
HypoElasticDevStress::addParticleState(std::vector<const VarLabel*>& ,
                                       std::vector<const VarLabel*>& ){}

void 
HypoElasticDevStress::initializeInternalVars( ParticleSubset* ,
                                              DataWarehouse* ) {}

void 
HypoElasticDevStress::getInternalVars( ParticleSubset*,
                                       DataWarehouse* ) {}

void 
HypoElasticDevStress::allocateAndPutInternalVars( ParticleSubset* ,
                                                  DataWarehouse* ) {}
void
HypoElasticDevStress::allocateAndPutRigid( ParticleSubset* ,
                                           DataWarehouse* ) {}

void HypoElasticDevStress::updateInternalStresses( const particleIndex,
                                                   const Matrix3& ,
                                                   DeformationState* ,
                                                   const double) {}

void HypoElasticDevStress::rotateInternalStresses( const particleIndex,
                                                   const Matrix3& ){}
