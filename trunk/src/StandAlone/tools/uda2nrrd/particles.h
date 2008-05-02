#ifndef UDA2NRRD_PARTICLES_H
#define UDA2NRRD_PARTICLES_H

#include "particleData.h"

#include <Packages/Uintah/StandAlone/tools/uda2nrrd/Args.h>
#include <Packages/Uintah/StandAlone/tools/uda2nrrd/QueryInfo.h>

#include <stdlib.h>

#include <string>
#include <vector>

/////////////////////////////////////////////////////////////////////
//
// If 'data' is NULL, then the data is in the x, y, and z arrays. 
//

typedef vector<Matrix3> matrixVec;

struct ParticleDataContainer {
  ParticleDataContainer() :
    name( "name not set" ),
    data( NULL ),
    x( NULL ), y( NULL ), z( NULL ),
    numParticles( 0 )
  {
  }

  ParticleDataContainer( const string & theName, float * theData, int theNumParticles ) :
    name( theName ),
    data( theData ),
    x( NULL ), y( NULL ), z( NULL ),
    numParticles( theNumParticles )
  {
  }

  string       name;
  float      * data;
  float      * x, * y, * z;
  unsigned int numParticles;
  
  // Addition
  matrixVec* matrixRep; // matrix repository
  int type; // 1 -> Vectors, 2 -> Tensors
};

//
// handleParticleDataContainer()
// 
// Loops over all patches and extracts an array of doubles corresponding to the
// requested data.
//
// The returned vector<> is of size 1 for most particle values, but is of size
// 3 for p.x (Point particle values).
//

template<class PartT>
ParticleDataContainer
handleParticleData( QueryInfo & qinfo, int matlNo, bool matlClassfication );

void
saveParticleData( std::vector<ParticleDataContainer> & data,
                  const std::string                  & filename,
				  variables & varColln );

#endif // UDA2NRRD_PARTICLES_H
