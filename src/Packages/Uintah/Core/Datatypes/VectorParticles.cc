#include "VectorParticles.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
using std::vector;
namespace Uintah {
namespace Datatypes {


using Uintah::DataArchive;
using Uintah::ParticleVariable;

using SCICore::Datatypes::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;


static Persistent* maker()
{
    return scinew VectorParticles;
}

PersistentTypeID VectorParticles::type_id("VectorParticles", "ParticleSet", maker);
#define VectorParticles_VERSION 3
void VectorParticles::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("VectorParticles::io(Piostream&)");
}

VectorParticles::VectorParticles()
  : have_minmax(false), psetH(0)
{
}

VectorParticles::VectorParticles(
		 const vector <ParticleVariable<Vector> >& vectors,
		 PSet* pset) :
  vectors(vectors),  psetH(pset), have_minmax(false)
{
}


VectorParticles::~VectorParticles()
{
}


void VectorParticles:: AddVar( const ParticleVariable<Vector> parts )
{
  vectors.push_back( parts );
}



void VectorParticles::compute_minmax()
{
  if( have_minmax )
    return;

  double min = 1e30, max = -1e30;
  vector<ParticleVariable<Vector> >::iterator it;
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

} // end namespace Datatypes
} // end namespace Uintah
