#include "TensorParticles.h"
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>

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
{
}

TensorParticles::TensorParticles(const ParticleVariable<Point>& positions,
				 const ParticleVariable<Matrix3>& tensors,
				 void* callbackClass) :
  positions(positions), tensors(tensors), cbClass(callbackClass)
{
}


TensorParticles::~TensorParticles()
{
}


void TensorParticles::compute_bounds()
{
  if( have_bounds )
    return;

  Point min(1e30,1e30,1e30), max(-1e30,-1e30,-1e30);

  ParticleSubset *ps = positions.getParticleSubset();
  for(ParticleSubset::iterator iter = ps->begin();
      iter != ps->end(); iter++){
    max = SCICore::Geometry::Max(positions[ *iter ], max);
    min = SCICore::Geometry::Min(positions[ *iter ], min);
  }

  if (min == max) {
    min = Point(0,0,0);
    max = Point(1,1,1);
  }
  have_bounds = true;
  bmin = min;
  bmax = max;
}

void TensorParticles::get_bounds(Point& p0, Point& p1)
{
  if( !have_bounds)
    compute_bounds();

  p0 = bmin;
  p1 = bmax;
}

void TensorParticles::compute_minmax()
{
  if( have_minmax )
    return;

  double min = 1e30, max = -1e30;
  ParticleSubset *ps = positions.getParticleSubset();
  for(ParticleSubset::iterator iter = ps->begin();
      iter != ps->end(); iter++){
    max = ( tensors[ *iter ].Norm() > max ) ?
      tensors[ *iter ].Norm() : max;
    min = ( tensors[ *iter ].Norm() < min ) ?
      tensors[ *iter ].Norm() : min;
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
} // end namespace Kurt
