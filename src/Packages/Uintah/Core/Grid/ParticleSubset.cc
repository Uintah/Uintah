
#include <Packages/Uintah/Core/Grid/ParticleSubset.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <algorithm>
#include <iostream>

using namespace Uintah;
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
    : d_pset(pset), d_matlIndex(matlIndex), d_patch(patch),
      d_gtype(Ghost::None), d_numGhostCells(0)
{
  init();
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
			       Ghost::GhostType gtype, int numGhostCells,
			       const vector<const Patch*>& neighbors,
			       const vector<ParticleSubset*>& neighbor_subsets)
  : d_pset(pset), d_matlIndex(matlIndex), d_patch(patch),
    d_gtype(gtype), d_numGhostCells(numGhostCells),
    neighbors(neighbors), neighbor_subsets(neighbor_subsets)
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
    SCI_THROW(InternalError("particleID variable must be ParticleVariable<long64>"));
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
    SCI_THROW(InternalError("ParticleSubsets should not be resized after creation"));
  d_allocatedSize = d_numParticles = newSize;
  d_particles = scinew particleIndex[newSize];
}

void
ParticleSubset::expand(particleIndex amount)
{
  if(amount < 10000)
    amount = 10000;
  d_allocatedSize += amount;
  if(d_numExpansions++ > 0){
    static bool warned = false;
    if(!warned){
      cerr << "Performance warning in ParticleSubset: " << d_numExpansions << " expansions occured\n";
      cerr << "Talk to Steve about a potential performance hit\n";
      cerr << "This message will only appear once per processor\n";
      cerr << "size=" << d_allocatedSize << ", numparticles=" << d_numParticles << '\n';
      warned = true;
    }
  }
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
