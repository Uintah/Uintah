
#include <Uintah/Grid/ParticleVariableBase.h>
#include <Uintah/Grid/TypeDescription.h>
#include <iostream>
using namespace Uintah;
using namespace std;

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

void ParticleVariableBase::getMPIBuffer(void*& buf, int& count,
					MPI_Datatype& datatype, bool& free_datatype,
					ParticleSubset* sendset)
{
   buf = getBasePointer();
   const TypeDescription* td = virtualGetTypeDescription()->getSubType();
   bool linear=true;
   ParticleSubset::iterator iter = sendset->begin();
   if(iter != sendset->end()){
      particleIndex last = *iter;
      for(;iter != sendset->end(); iter++){
	 particleIndex idx = *iter;
	 if(idx != last+1){
	    linear=false;
	    break;
	 }
      }
   }
   if(linear){
      datatype=td->getMPIType();
      count = sendset->getParticleSet()->numParticles();
   } else {
      vector<int> blocklens(sendset->numParticles(), 1);
      MPI_Type_indexed(sendset->numParticles(), &blocklens[0],
		       sendset->begin(), td->getMPIType(), &datatype);
      MPI_Type_commit(&datatype);
      count=1;
      free_datatype=true;
   } 
}

//
// $Log$
// Revision 1.4.4.3  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.4.4.2  2000/10/07 00:04:44  witzel
// using namespace std;
//
// Revision 1.4.4.1  2000/10/02 15:00:45  sparker
// Support for sending only boundary particles
//
// Revision 1.4  2000/07/27 22:39:50  sparker
// Implemented MPIScheduler
// Added associated support
//
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

