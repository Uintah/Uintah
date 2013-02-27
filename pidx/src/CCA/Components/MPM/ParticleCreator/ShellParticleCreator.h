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

#ifndef __SHELL_PARTICLE_CREATOR_H__
#define __SHELL_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

/*******************************************************************************

CLASS

   ShellParticleCreator
   
   Creates particles that have properties of shells


GENERAL INFORMATION

   ShellParticleCreator.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Shell particles

DESCRIPTION
   
   Same as membrane particle creator except that a thickness parameter
   is attached to shells along with the corotational tangents to the
   shell at the particle.  The directions of the directors of the shell 
   are calculated using the tangents.

WARNING
  
*******************************************************************************/
  class ShellParticleCreator : public ParticleCreator {
  public:
    
    /////////////////////////////////////////////////////////////////////////
    //
    // Constructor and destructor
    //
    ShellParticleCreator(MPMMaterial* matl, 
                         MPMFlags* flags);

    virtual ~ShellParticleCreator();

    /////////////////////////////////////////////////////////////////////////
    //
    // Actually create particles using geometry
    //
    virtual ParticleSubset* createParticles(MPMMaterial* matl, 
                                            particleIndex numParticles,
                                            CCVariable<short int>& cellNAPID,
                                            const Patch*, 
                                            DataWarehouse* new_dw,
                                            vector<GeometryObject*>&);

    /////////////////////////////////////////////////////////////////////////
    //
    // Count particles in a patch
    //
    virtual particleIndex countParticles(const Patch*,
                                         std::vector<GeometryObject*>&) ;
    virtual particleIndex countAndCreateParticles(const Patch*,
                                                  GeometryObject* obj) ;

    /////////////////////////////////////////////////////////////////////////
    //
    // Make sure data is copied when particles cross patch boundaries
    //
    virtual void registerPermanentParticleState(MPMMaterial* matl);
                                                

  };

} // End of namespace Uintah

#endif // __SHELL_PARTICLE_CREATOR_H__
