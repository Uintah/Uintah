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

ParticleSubset::ParticleSubset(ParticleSet* pset)
    : d_pset(pset)
{
    d_pset->addReference();
    int np = d_pset->numParticles();
    d_particles.resize(np);
    for(int i=0;i<np;i++)
	d_particles[i]=i;
}

//
// $Log$
// Revision 1.3  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
