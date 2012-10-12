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

#ifndef __GUV_PARTICLE_CREATOR_H__
#define __GUV_PARTICLE_CREATOR_H__

#include "ShellParticleCreator.h"

namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! \class GUVParticleCreator
    \brief Creates particles that have properties of GUVs
    \author Biswajit Banerjee \n
    Department of Mechanical Engineering \n
    University of Utah \n
    Center for the Simulation of Accidental Fires and Explosions \n

    Same as membrane particle creator except that a thickness parameter
    is attached to GUVs along with the corotational tangents to the
    GUV at the particle.  The directions of the directors of the GUV 
    are calculated using the tangents.
  */
  ///////////////////////////////////////////////////////////////////////////

  class GUVParticleCreator : public ShellParticleCreator {
  public:
    
    /////////////////////////////////////////////////////////////////////////
    //
    // Constructor and destructor
    //
    GUVParticleCreator(MPMMaterial* matl, 
                       MPMLabel* lb,
                       MPMFlags* flags);
    virtual ~GUVParticleCreator();
    
    /////////////////////////////////////////////////////////////////////////
    //
    // Actually create particles using geometry
    //
    virtual ParticleSubset* createParticles(MPMMaterial* matl, 
                                            particleIndex numParticles,
                                            CCVariable<short int>& cellNAPID,
                                            const Patch*, 
                                            DataWarehouse* new_dw,
                                            MPMLabel* lb,
                                            vector<GeometryObject*>&);

    /////////////////////////////////////////////////////////////////////////
    //
    // Create and count particles in a patch
    //
    virtual particleIndex countAndCreateParticles(const Patch*,
                                                  GeometryObject* obj) ;

  protected:

    typedef map<pair<const Patch*,GeometryObject*>,vector<int> > geomint;
    geompoints d_pos;
    geomvols d_vol;
    geomint d_type;
    geomvols d_thick;
    geomvecs d_norm;
  };

} // End of namespace Uintah

#endif // __GUV_PARTICLE_CREATOR_H__
