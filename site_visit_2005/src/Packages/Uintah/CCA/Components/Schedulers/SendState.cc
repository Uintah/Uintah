
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSubset.h>
#include <Packages/Uintah/Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>


using namespace Uintah;
using namespace SCIRun;

SendState::SendState()
{
}

SendState::~SendState()
{
  for(maptype::iterator iter = sendSubsets.begin();
      iter != sendSubsets.end();iter++)
    delete iter->second;
}

ParticleSubset*
SendState::find_sendset(int dest, const Patch* patch, int matlIndex,
                        Ghost::GhostType gt /*=Ghost::None*/,
                        int numgc /*=0*/, int dwid /* =0 */) const
{
  maptype::const_iterator iter = 
    sendSubsets.find( make_pair( PSPatchMatlGhost(patch, matlIndex, gt, numgc, dwid), dest) );
  if(iter == sendSubsets.end())
    return 0;
  return iter->second;
}

void
SendState::add_sendset(ParticleSubset* sendset, int dest, const Patch* patch, 
                       int matlIndex, Ghost::GhostType gt /*=Ghost::None*/,
                       int numgc /*=0*/, int dwid /*=0*/)
{
  maptype::iterator iter = 
    sendSubsets.find(make_pair(PSPatchMatlGhost(patch,matlIndex,gt,numgc,dwid), dest));
  if(iter != sendSubsets.end())
    SCI_THROW(InternalError("sendSubset already exists"));
  sendSubsets[make_pair(PSPatchMatlGhost(patch, matlIndex, gt, numgc, dwid), dest)]=sendset;
}

void SendState::print() 
{
  //cout << Parallel::getMPIRank() << " SENDSETS: " << endl;
  for (maptype::iterator iter = sendSubsets.begin(); iter != sendSubsets.end(); iter++) {
    //cout << Parallel::getMPIRank() << ' ' << *(iter->second) << " src/dest: " 
    //     << iter->first.second << endl;
    
  }
}
