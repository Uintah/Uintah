/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Datatypes/VectorParticles.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Grid.h>

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

