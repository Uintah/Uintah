/* REFERENCED */
static char *id="@(#) $Id$";

#include "ParticleSubset.h"
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;
using namespace std;

ParticleSubset::~ParticleSubset()
{
   if(d_pset && d_pset->removeReference())
      delete d_pset;
   for(int i=0;i<neighbor_subsets.size();i++)
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
   for(int i=0;i<neighbor_subsets.size();i++)
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

//
// $Log$
// Revision 1.10  2000/08/21 23:27:07  sparker
// Added getReferenceCount() method to RefCounted
// Correctly maintain ref counts on neighboring particle subsets in ParticleSubset
//
// Revision 1.9  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.8  2000/06/15 21:57:18  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.7  2000/05/30 20:19:30  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.6  2000/05/20 08:09:24  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.5  2000/05/20 02:36:06  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.4  2000/05/10 20:03:01  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.3  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
