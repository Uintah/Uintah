
#include <Packages/Uintah/Core/Grid/ParticleVariableBase.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>

#include <Core/Thread/Mutex.h>

#include <iostream>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

extern Mutex MPITypeLock;

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

void ParticleVariableBase::getMPIBuffer(BufferInfo& buffer,
					ParticleSubset* sendset)
{
  const TypeDescription* td = virtualGetTypeDescription()->getSubType();

  //  cerr << "ParticleVariableBase::getMPIBuffer for a " <<  td->getName() 
  //       << endl;
  //  cerr << "   buffer: " << &buffer << ", sendset: " << sendset << "\n";

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
  void* buf = getBasePointer();
  int count = sendset->numParticles();
  if(linear){
    buffer.add(buf, count, td->getMPIType(), false);
  } else {
    vector<int> blocklens( count, 1);
    MPI_Datatype datatype;

    //    cerr << "cnt: " << count << ", buf: " << buf << "\n";
   MPITypeLock.lock();
    MPI_Type_indexed(count, &blocklens[0],
		     sendset->getPointer(), td->getMPIType(), &datatype);
    MPI_Type_commit(&datatype);
   MPITypeLock.unlock();
    
    buffer.add(buf, 1, datatype, true);
  } 
}

