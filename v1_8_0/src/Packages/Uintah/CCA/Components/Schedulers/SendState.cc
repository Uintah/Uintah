
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/Core/Grid/ParticleSubset.h>
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
SendState::find_sendset(const Patch* patch, int matlIndex, int dest) const
{
  maptype::const_iterator iter = 
    sendSubsets.find( make_pair( make_pair(patch, matlIndex), dest ) );
  if(iter == sendSubsets.end())
    return 0;
  return iter->second;
}

void
SendState::add_sendset(const Patch* patch, int matlIndex, int dest,
		       ParticleSubset* sendset)
{
  maptype::iterator iter = sendSubsets.find(make_pair(make_pair(patch,
								matlIndex),
						      dest));
  if(iter != sendSubsets.end())
    SCI_THROW(InternalError("sendSubset already exists"));
  sendSubsets[make_pair(make_pair(patch, matlIndex), dest)]=sendset;
}
