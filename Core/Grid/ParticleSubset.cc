
#include <Packages/Uintah/Core/Grid/ParticleSubset.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;

ParticleSubset::~ParticleSubset()
{
   if(d_pset && d_pset->removeReference())
      delete d_pset;
   for(int i=0;i<(int)neighbor_subsets.size();i++)
      if(neighbor_subsets[i]->removeReference())
	 delete neighbor_subsets[i];
}

ParticleSubset::ParticleSubset() :
  d_pset( scinew ParticleSet )
{
   d_pset->addReference();
}

ParticleSubset::ParticleSubset(ParticleSet* pset, bool fill,
			       int matlIndex, const Patch* patch)
    : d_pset(pset), d_matlIndex(matlIndex), d_patch(patch),
      d_gtype(Ghost::None), d_numGhostCells(0)
{
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
   d_pset->addReference();
   for(int i=0;i<(int)neighbor_subsets.size();i++)
      neighbor_subsets[i]->addReference();
   if(fill)
      fillset();
}

void
ParticleSubset::fillset()
{
   int np = d_pset->numParticles();
   d_particles.resize(np);
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
    throw InternalError("particleID variable must be ParticleVariable<long64>");
  compareIDFunctor comp(pIDs);
  ::sort(d_particles.begin(), d_particles.end(), comp);
}

