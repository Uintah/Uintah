#include <Packages/Uintah/Core/Datatypes/VectorParticles.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>

#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>

using std::vector;

namespace Uintah {

using namespace SCIRun;


static Persistent* maker()
{
    return scinew VectorParticles;
}

PersistentTypeID VectorParticles::type_id("VectorParticles", "ParticleSet", maker);

#define VectorParticles_VERSION 3
void VectorParticles::io(Piostream&)
{
    NOT_FINISHED("VectorParticles::io(Piostream&)");
}

VectorParticles::VectorParticles()
  : have_minmax(false), psetH(0)
{
}

VectorParticles::VectorParticles(
		 const vector<ShareAssignParticleVariable<Vector> >& vectors,
		 PSet* pset) :
  have_minmax(false), psetH(pset), vectors(vectors)
{
}


VectorParticles::~VectorParticles()
{
}


void VectorParticles:: AddVar( const ParticleVariable<Vector>& parts )
{
  vectors.push_back( parts );
}


void VectorParticles::compute_minmax()
{
  if( have_minmax )
    return;

  double min = 1e30, max = -1e30;
  vector<ShareAssignParticleVariable<Vector> >::iterator it;
  for( it = vectors.begin(); it != vectors.end(); it++){
    ParticleSubset *ps = (*it).getParticleSubset();
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      max = ( (*it)[ *iter ].length() > max ) ?
	(*it)[ *iter ].length() : max;
      min = ( (*it)[ *iter ].length() < min ) ?
	(*it)[ *iter ].length() : min;
    }
  }
  if (min == max) {
    min -= 0.001;
    max += 0.001;
  }
  have_minmax = true;
  data_min = min;
  data_max = max;
}

void VectorParticles::get_minmax(double& v0, double& v1)
{
  if(!have_minmax)
    compute_minmax();

  v0 = data_min;
  v1 = data_max; 
}

} // End namespace Uintah

