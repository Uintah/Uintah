
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/Core/Grid/ParticleSubset.h>

using namespace Uintah;

SendState::SendState()
{
}

SendState::~SendState()
{
    for(std::map<std::pair<std::pair<const Patch*, int>, int>, ParticleSubset*>::iterator iter = d_sendSubsets.begin();iter != d_sendSubsets.end();iter++){
       delete iter->second;
    }
}


