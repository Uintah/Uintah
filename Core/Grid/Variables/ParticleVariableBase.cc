/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Core/Grid/Variables/ParticleVariableBase.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Parallel/BufferInfo.h>

#include <Core/Thread/Mutex.h>

#include <iostream>

using namespace std;

using namespace Uintah;
using namespace SCIRun;


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
    MPI_Type_indexed( count, &blocklens[0],
                        sendset->getPointer(), td->getMPIType(), &datatype );
    MPI_Type_commit(&datatype);
    
    buffer.add(buf, 1, datatype, true);
  } 
}

void ParticleVariableBase::setParticleSubset(ParticleSubset* subset)
{
  if(d_pset && d_pset->removeReference())
    delete d_pset;
  d_pset = subset;
  if(d_pset)
    d_pset->addReference();
}
