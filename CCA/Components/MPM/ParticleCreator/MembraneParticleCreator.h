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

#ifndef __MEMBRANE_PARTICLE_CREATOR_H__
#define __MEMBRANE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class MembraneParticleCreator : public ParticleCreator {
  public:
    
    MembraneParticleCreator(MPMMaterial* matl,MPMFlags* flags);
    virtual ~MembraneParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl, 
                                            particleIndex numParticles,
                                            CCVariable<short int>& cellNAPID,
                                            const Patch*, 
                                            DataWarehouse* new_dw,
                                            vector<GeometryObject*>&);

    virtual particleIndex countParticles(const Patch*,
                                         std::vector<GeometryObject*>&) ;
    virtual particleIndex countAndCreateParticles(const Patch*,
                                                  GeometryObject* obj) ;

    virtual void registerPermanentParticleState(MPMMaterial* matl);

    const VarLabel* pTang1Label;
    const VarLabel* pTang2Label;
    const VarLabel* pNormLabel;
    const VarLabel* pTang1Label_preReloc;
    const VarLabel* pTang2Label_preReloc;
    const VarLabel* pNormLabel_preReloc;


  protected:
    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
                                              int dwi, const Patch* patch,
                                              DataWarehouse* new_dw);
    

    ParticleVariable<Vector> pTang1, pTang2, pNorm;


        
  };



} // End of namespace Uintah

#endif // __MEMBRANE_PARTICLE_CREATOR_H__
