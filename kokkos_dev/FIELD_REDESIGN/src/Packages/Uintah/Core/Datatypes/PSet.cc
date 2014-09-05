#include "PSet.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
using std::vector;
namespace Uintah {
namespace Datatypes {


using Uintah::ParticleVariable;
using SCICore::Datatypes::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;

static Persistent* maker()
{
    return scinew PSet;
}

PersistentTypeID PSet::type_id("PSet", "ParticleSet", maker);
#define PSet_VERSION 3
void PSet::io(Piostream&)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Geometry::Pio;
    NOT_FINISHED("PSet::io(Piostream&)");
}

PSet::PSet()
  : have_bounds(false)
{
}

PSet::PSet( const vector <ParticleVariable<Point> >& positions,
	    const vector <ParticleVariable<long> >& ids,
	    const vector <const Patch *> patches,
	    void* callbackClass):
  positions(positions), particle_ids(ids),
  patches( patches ), cbClass(callbackClass),
  have_bounds(false)
{
}


PSet::~PSet()
{
}

void PSet:: AddParticles( const ParticleVariable<Point> locs,
			  const ParticleVariable<long>  ids,
			  const Patch* patch)
{
  positions.push_back( locs );
  particle_ids.push_back( ids );
  patches.push_back( patch);
}

void PSet::get_bounds(Point& p0, Point& p1)
{
  if( !have_bounds)
    compute_bounds();

  p0 = bmin;
  p1 = bmax;
}

void PSet::compute_bounds()
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



} // end namespace Datatypes
} // end namespace Uintah
