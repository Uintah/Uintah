
#include "VectorParticles.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>

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
{
}

VectorParticles::VectorParticles(
			 const vector <ParticleVariable<Point> >& positions,
			 const vector <ParticleVariable<Vector> >& vectors,
			 void* callbackClass) :
  positions(positions), vectors(vectors), cbClass(callbackClass),
  have_bounds(false), have_minmax(false)
{
}


VectorParticles::~VectorParticles()
{
}
void 
VectorParticles:: AddVar( const ParticleVariable<Point> locs,
			  const ParticleVariable<Vector> parts,
			  const Patch*)
{
  positions.push_back( locs );
  vectors.push_back( parts );
}

void VectorParticles::compute_bounds()
{
  if( have_bounds )
    return;

  Point min(1e30,1e30,1e30), max(-1e30,-1e30,-1e30);

  vector<ParticleVariable<Point> >::iterator it;
  for( it = positions.begin(); it != positions.end(); it++){
    ParticleSubset *ps = (*it).getParticleSubset();
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      max = SCICore::Geometry::Max((*it)[ *iter ], max);
      min = SCICore::Geometry::Min((*it)[ *iter ], min);
    }
  }
  if (min == max) {
    min = Point(0,0,0);
    max = Point(1,1,1);
  }
  have_bounds = true;
  bmin = min;
  bmax = max;
}

void VectorParticles::get_bounds(Point& p0, Point& p1)
{
  if( !have_bounds)
    compute_bounds();

  p0 = bmin;
  p1 = bmax;
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
} // end namespace Kurt
