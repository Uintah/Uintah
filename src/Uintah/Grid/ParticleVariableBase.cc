#include <Uintah/Grid/ParticleVariableBase.h>

using namespace Uintah;

ParticleVariableBase::~ParticleVariableBase()
{
   if(d_pset && d_pset->removeReference())
      delete d_pset;
}

ParticleVariableBase::ParticleVariableBase(ParticleSubset* pset)
   : d_pset(pset)
{
   if(d_pset)
      d_pset->addReference();
}

ParticleVariableBase::ParticleVariableBase(const ParticleVariableBase& copy)
   : d_pset(copy.d_pset)
{
   if(d_pset)
      d_pset->addReference();
}   

ParticleVariableBase& ParticleVariableBase::operator=(const ParticleVariableBase& copy)
{
   if(this != &copy){
      if(d_pset && d_pset->removeReference())
	 delete d_pset;
      d_pset = copy.d_pset;
      if(d_pset)
	 d_pset->addReference();
   }
   return *this;
}   

//
// $Log$
// Revision 1.3  2000/06/15 21:57:19  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.2  2000/04/26 06:48:52  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/20 22:58:20  sparker
// Resolved undefined symbols
// Trying to make stuff work
//
//

