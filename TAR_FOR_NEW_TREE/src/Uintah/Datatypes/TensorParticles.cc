#include "TensorParticles.h"
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
    return scinew TensorParticles;
}

PersistentTypeID TensorParticles::type_id("TensorParticles", "ParticleSet", maker);
#define TensorParticles_VERSION 3
void TensorParticles::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("TensorParticles::io(Piostream&)");
}

TensorParticles::TensorParticles()
  : have_minmax(false), psetH(0)
{
}

TensorParticles::TensorParticles(
		 const vector <ParticleVariable<Matrix3> >& tensors,
		 PSet* pset) :
  tensors(tensors),  psetH(pset), have_minmax(false)
{
}


TensorParticles::~TensorParticles()
{
}


void TensorParticles:: AddVar( const ParticleVariable<Matrix3> parts )
{
  tensors.push_back( parts );
}


void TensorParticles::compute_minmax()
{
  if( have_minmax )
    return;

  double min = 1e30, max = -1e30;
  vector<ParticleVariable<Matrix3> >::iterator it;
  for( it = tensors.begin(); it != tensors.end(); it++){
    ParticleSubset *ps = (*it).getParticleSubset();
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      max = ( (*it)[ *iter ].Norm() > max ) ?
	(*it)[ *iter ].Norm() : max;
      min = ( (*it)[ *iter ].Norm() < min ) ?
	(*it)[ *iter ].Norm() : min;
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

void TensorParticles::get_minmax(double& v0, double& v1)
{
  if(!have_minmax)
    compute_minmax();

  v0 = data_min;
  v1 = data_max; 
}
} // end namespace Datatypes
} // end namespace Uintah
