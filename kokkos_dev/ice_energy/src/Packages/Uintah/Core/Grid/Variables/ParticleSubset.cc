
#include <Packages/Uintah/Core/Grid/Variables/ParticleSubset.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/ProgressiveWarning.h>

#include <sgi_stl_warnings_off.h>
#include <algorithm>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

ParticleSubset::~ParticleSubset()
{
  if(d_pset && d_pset->removeReference())
    delete d_pset;
  for(int i=0;i<(int)neighbor_subsets.size();i++)
    if(neighbor_subsets[i]->removeReference())
      delete neighbor_subsets[i];
  if(d_particles)
    delete[] d_particles;
}

ParticleSubset::ParticleSubset()
 : d_pset( scinew ParticleSet )
{
  init();
  d_pset->addReference();
}

ParticleSubset::ParticleSubset(ParticleSet* pset, bool fill,
			       int matlIndex, const Patch* patch,
			       particleIndex sizeHint)
    : d_pset(pset), d_matlIndex(matlIndex), d_patch(patch)
{
  init();

  if (patch) {
    d_low = patch->getLowIndex();
    d_high = patch->getHighIndex();
  }
  else {
    // don't matter...
    d_low = IntVector(0,0,0);
    d_high = IntVector(0,0,0);
  }
  if(sizeHint != 0){
    d_allocatedSize = sizeHint;
    d_particles = scinew particleIndex[d_allocatedSize];
  }
  d_pset->addReference();
  if(fill)
    fillset();
}

ParticleSubset::ParticleSubset(ParticleSet* pset, bool fill,
			       int matlIndex, const Patch* patch,
                               IntVector low, IntVector high,
			       const vector<const Patch*>& neighbors,
			       const vector<ParticleSubset*>& neighbor_subsets)
  : d_pset(pset), d_matlIndex(matlIndex), d_patch(patch),
    d_low(low), d_high(high),
    neighbors(neighbors), neighbor_subsets(neighbor_subsets)
{
  init();
  d_pset->addReference();
  for(int i=0;i<(int)neighbor_subsets.size();i++)
    neighbor_subsets[i]->addReference();
  if(fill)
    fillset();
}

ParticleSubset::ParticleSubset(ParticleSet* pset, bool fill,
                               int matlIndex, const Patch* patch,
                               IntVector low, IntVector high,
                               particleIndex /*sizeHint*/)
  : d_pset(pset), d_matlIndex(matlIndex), d_patch(patch),
    d_low(low), d_high(high)
{
  init();
  d_pset->addReference();
  for(int i=0;i<(int)neighbor_subsets.size();i++)
    neighbor_subsets[i]->addReference();
  if(fill)
    fillset();
}

void
ParticleSubset::fillset()
{
  if(d_particles)
    delete[] d_particles;
  int np = d_numParticles = d_pset->numParticles();
  d_particles = scinew particleIndex[d_numParticles];
  for(int i=0;i<np;i++)
    d_particles[i]=i;
}


class compareIDFunctor
{
public:
  compareIDFunctor(ParticleVariable<long64>* particleIDs)
    : particleIDs_(particleIDs) {}
  
  bool operator()(particleIndex x, particleIndex y)
  {
    return (*particleIDs_)[x] < (*particleIDs_)[y];
  }

private:
  ParticleVariable<long64>* particleIDs_;
};

void
ParticleSubset::sort(ParticleVariableBase* particleIDs)
{
  ParticleVariable<long64>* pIDs =
    dynamic_cast<ParticleVariable<long64>*>(particleIDs);
  if (pIDs == 0)
    SCI_THROW(InternalError("particleID variable must be ParticleVariable<long64>", __FILE__, __LINE__));
  compareIDFunctor comp(pIDs);
  ::sort(d_particles, d_particles+d_numParticles, comp);
}

void
ParticleSubset::init()
{
  d_particles = 0;
  d_numParticles = 0;
  d_allocatedSize = 0;
  d_numExpansions = 0;
}

void
ParticleSubset::resize(particleIndex newSize)
{
  // Check for spurious resizes
  if(d_particles)
    SCI_THROW(InternalError("ParticleSubsets should not be resized after creation", __FILE__, __LINE__));
  d_allocatedSize = d_numParticles = newSize;
  d_particles = scinew particleIndex[newSize];
}

void
ParticleSubset::expand(particleIndex amount)
{
  particleIndex minAmount = d_numParticles>>2;
  if(minAmount < 10)
    minAmount = 10;
  if(amount < minAmount)
    amount = minAmount;
  d_allocatedSize += minAmount;
#if 0
  if(d_numExpansions++ > 18){
    static ProgressiveWarning warn("Performance warning in ParticleSubset",10);
    warn.changeMessage("Performance warning in ParticleSubset, more than 18 expansions occured");
    warn.invoke();
  }
#endif
  particleIndex* newparticles = scinew particleIndex[d_allocatedSize];
  if(d_particles){
    for(particleIndex i = 0; i < d_numParticles; i++)
      newparticles[i] = d_particles[i];
    delete[] d_particles;
  }
  d_particles = newparticles;
}

particleIndex ParticleSubset::addParticles(particleIndex count)
{
  particleIndex oldsize = d_pset->addParticles(count);
  particleIndex newsize = oldsize+count;
  ASSERTEQ(oldsize, d_numParticles);
  particleIndex start = oldsize;

  d_allocatedSize = newsize;
  particleIndex* newparticles = scinew particleIndex[d_allocatedSize];
  if(d_particles){
    for(particleIndex i = 0; i < oldsize; i++)
      newparticles[i] = d_particles[i];
    delete[] d_particles;
  }
  d_particles = newparticles;
  d_numParticles = newsize;

  for(particleIndex idx = oldsize; idx < newsize; idx++, start++)
    d_particles[start] = idx;
  return oldsize;  // The beginning of the new index range
}

ostream& operator<<(ostream& out, ParticleSubset& pset)
{
    out << &pset
        << " patch: " << pset.getPatch() << " (" << (pset.getPatch()?pset.getPatch()->getID():0)
        << "), matl "
        << pset.getMatlIndex() << " range [" << pset.getLow() 
        << ", " << pset.getHigh() << ") " 
        << pset.numParticles() << " particles, " 
        << pset.getNeighbors().size() << " neighbors" ;
    return out;
}
