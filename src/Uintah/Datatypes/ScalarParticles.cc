#include "ScalarParticles.h"
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
    return scinew ScalarParticles;
}

PersistentTypeID ScalarParticles::type_id("ScalarParticles", "ParticleSet", maker);
#define ScalarParticles_VERSION 3
void ScalarParticles::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("ScalarParticles::io(Piostream&)");
}

ScalarParticles::ScalarParticles()
  : have_bounds(false), have_minmax(false)
{
}

ScalarParticles::ScalarParticles(
		 const vector <ParticleVariable<Point> >& positions,
		 const vector <ParticleVariable<double> >& scalars,
		 void* callbackClass):
  positions(positions), scalars(scalars),  cbClass(callbackClass),
  have_bounds(false), have_minmax(false)
{
}


ScalarParticles::~ScalarParticles()
{
}

void ScalarParticles::compute_bounds()
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

void ScalarParticles:: AddVar( const ParticleVariable<Point> locs,
			       const ParticleVariable<double> parts,
			       const Patch*)
{
  positions.push_back( locs );
  scalars.push_back( parts );
}

void ScalarParticles::get_bounds(Point& p0, Point& p1)
{
  if( !have_bounds)
    compute_bounds();

  p0 = bmin;
  p1 = bmax;
}

void ScalarParticles::compute_minmax()
{
  if( have_minmax )
    return;

  double min = 1e30, max = -1e30;
  vector<ParticleVariable<double> >::iterator it;
  for( it = scalars.begin(); it != scalars.end(); it++){
    
    ParticleSubset *ps = (*it).getParticleSubset();
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      max = ( (*it)[ *iter ] > max ) ?
	(*it)[ *iter ] : max;
      min = ( (*it)[ *iter ] < min ) ?
	(*it)[ *iter ] : min;
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

void ScalarParticles::get_minmax(double& v0, double& v1)
{
  if(!have_minmax)
    compute_minmax();

  v0 = data_min;
  v1 = data_max; 
}

} // end namespace Datatypes
} // end namespace Kurt
