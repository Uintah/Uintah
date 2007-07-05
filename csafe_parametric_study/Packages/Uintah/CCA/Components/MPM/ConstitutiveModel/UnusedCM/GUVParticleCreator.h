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
    Copyright (C) 2004 University of Utah 

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
