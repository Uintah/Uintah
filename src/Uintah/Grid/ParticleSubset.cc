/* REFERENCED */
static char *id="@(#) $Id$";

#include "ParticleSubset.h"

#include <iostream>

using namespace Uintah;
using namespace std;

ParticleSubset::~ParticleSubset()
{
    if(d_pset && d_pset->removeReference())
	delete d_pset;
}

ParticleSubset::ParticleSubset(ParticleSet* pset, bool fill)
    : d_pset(pset)
{
   d_pset->addReference();
   if(fill){
      int np = d_pset->numParticles();
      d_particles.resize(np);
      for(int i=0;i<np;i++)
	 d_particles[i]=i;
   }
}

//
// $Log$
// Revision 1.4  2000/05/10 20:03:01  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.3  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
