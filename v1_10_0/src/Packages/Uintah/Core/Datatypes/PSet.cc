#include <Packages/Uintah/Core/Datatypes/PSet.h>
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
    return scinew PSet;
}

PersistentTypeID PSet::type_id("PSet", "ParticleSet", maker);
#define PSet_VERSION 3
void PSet::io(Piostream&)
{
    NOT_FINISHED("PSet::io(Piostream&)");
}

PSet::PSet()
  : have_bounds(false)
{
}

PSet::PSet( const vector <ShareAssignParticleVariable<Point> >& positions,
	    const vector <ShareAssignParticleVariable<long64> >& ids,
	    const vector <const Patch *> patches,
	    void* callbackClass):
  have_bounds( false ),
  cbClass( callbackClass ),
  positions( positions ), particle_ids( ids ),
  patches( patches )
{
}


PSet::~PSet()
{
}

void PSet:: AddParticles( const ParticleVariable<Point>& locs,
			  const ParticleVariable<long64>&  ids,
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

  vector<ShareAssignParticleVariable<Point> >::iterator it;
  for( it = positions.begin(); it != positions.end(); it++){
    
    ParticleSubset *ps = (*it).getParticleSubset();
    for(ParticleSubset::iterator iter = ps->begin();
	iter != ps->end(); iter++){
      max = Max((*it)[ *iter ], max);
      min = Min((*it)[ *iter ], min);
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

} // End namespace Uintah
