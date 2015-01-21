
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Disclosure/TypeUtils.h>
#include <SCIRun/Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <SCIRun/Core/Util/ProgressiveWarning.h>

#include <sgi_stl_warnings_off.h>
#include <algorithm>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

ParticleSubset::~ParticleSubset()
{
  for(int i=0;i<(int)neighbor_subsets.size();i++)
    if(neighbor_subsets[i]->removeReference())
      delete neighbor_subsets[i];
  if(d_particles)
    delete[] d_particles;
}

ParticleSubset::ParticleSubset() : d_numParticles(0)
{
  init();
}

ParticleSubset::ParticleSubset(int num_particles, int matlIndex, const Patch* patch)
    : d_numParticles(num_particles), d_matlIndex(matlIndex), d_patch(patch)
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
  fillset();
}

ParticleSubset::ParticleSubset(int num_particles, int matlIndex, const Patch* patch,
                               IntVector low, IntVector high,
			       const vector<const Patch*>& neighbors,
			       const vector<ParticleSubset*>& neighbor_subsets)
  : d_numParticles(num_particles), d_matlIndex(matlIndex), d_patch(patch),
    d_low(low), d_high(high),
    neighbors(neighbors), neighbor_subsets(neighbor_subsets)
{
  init();
  for(int i=0;i<(int)neighbor_subsets.size();i++)
    neighbor_subsets[i]->addReference();
  fillset();
}

ParticleSubset::ParticleSubset(int num_particles, int matlIndex, const Patch* patch,
                               IntVector low, IntVector high)
  : d_numParticles(num_particles), d_matlIndex(matlIndex), d_patch(patch),
    d_low(low), d_high(high)
{
  init();
  for(int i=0;i<(int)neighbor_subsets.size();i++)
    neighbor_subsets[i]->addReference();
  fillset();
}

void
ParticleSubset::fillset()
{
  if (d_numParticles > 0) {
    d_particles = scinew particleIndex[d_numParticles];
    for(int i=0;i<d_numParticles;i++)
      d_particles[i]=i;
    d_allocatedSize = d_numParticles;
  }
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
  d_allocatedSize += amount;
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
  if(d_numParticles + count > d_allocatedSize)
    expand(count);


  particleIndex oldsize = d_numParticles;
  d_numParticles += count;

  for(particleIndex idx = oldsize; idx < d_numParticles; idx++)
    d_particles[idx] = idx;
  return oldsize;  // The beginning of the new index range
}

namespace Uintah {
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
} // end namespace Uintah
