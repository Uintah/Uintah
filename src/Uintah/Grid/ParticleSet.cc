/* REFERENCED */
static char *id="@(#) $Id$";

#include "ParticleSet.h"

using namespace Uintah;

ParticleSet::ParticleSet()
    : d_numParticles(0)
{
}

ParticleSet::ParticleSet(particleIndex numParticles)
    : d_numParticles(numParticles)
{
}

ParticleSet::~ParticleSet()
{
}

//
// $Log$
// Revision 1.4  2000/04/28 07:35:36  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.3  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
