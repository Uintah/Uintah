
#include "ParticleSubset.h"
#include <iostream>
using std::cerr;

ParticleSubset::~ParticleSubset()
{
    if(pset && pset->removeReference())
	delete pset;
}

ParticleSubset::ParticleSubset(ParticleSet* pset)
    : pset(pset)
{
    pset->addReference();
    int np = pset->numParticles();
    particles.resize(np);
    for(int i=0;i<np;i++)
	particles[i]=i;
}
