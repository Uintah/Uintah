
#include <CCA/Components/Schedulers/SendState.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Core/Parallel/Parallel.h>
#include <SCIRun/Core/Exceptions/InternalError.h>


using namespace Uintah;
using namespace SCIRun;

SendState::SendState()
{
}

SendState::~SendState()
{
  for(maptype::iterator iter = sendSubsets.begin();
      iter != sendSubsets.end();iter++) {
    if (iter->second->removeReference())
      delete iter->second;
  }
}

ParticleSubset*
SendState::find_sendset(int dest, const Patch* patch, int matlIndex,
                        IntVector low, IntVector high, int dwid /* =0 */) const
{
  maptype::const_iterator iter = 
    sendSubsets.find( make_pair( PSPatchMatlGhost(patch, matlIndex, low, high, dwid), dest) );
  if(iter == sendSubsets.end())
    return 0;
  return iter->second;
}

void
SendState::add_sendset(ParticleSubset* sendset, int dest, const Patch* patch, 
                       int matlIndex, IntVector low, IntVector high, int dwid /*=0*/)
{
  maptype::iterator iter = 
    sendSubsets.find(make_pair(PSPatchMatlGhost(patch,matlIndex,low,high,dwid), dest));
  if(iter != sendSubsets.end())
    SCI_THROW(InternalError("sendSubset already exists", __FILE__, __LINE__));
  sendSubsets[make_pair(PSPatchMatlGhost(patch, matlIndex, low, high, dwid), dest)]=sendset;
  sendset->addReference();
}

void SendState::print() 
{
  //cout << Parallel::getMPIRank() << " SENDSETS: " << endl;
  for (maptype::iterator iter = sendSubsets.begin(); iter != sendSubsets.end(); iter++) {
    //cout << Parallel::getMPIRank() << ' ' << *(iter->second) << " src/dest: " 
    //     << iter->first.second << endl;
    
  }
}
