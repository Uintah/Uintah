
#include <Uintah/Components/Schedulers/SendState.h>
#include <Uintah/Grid/ParticleSubset.h>

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

//
// $Log$
// Revision 1.2  2000/12/10 09:06:11  sparker
// Merge from csafe_risky1
//
// Revision 1.1.2.2  2000/10/02 17:33:39  sparker
// Fixed boundary particles code for multiple materials
// Free ParticleSubsets used for boundary particle sends
//
// Revision 1.1.2.1  2000/10/02 15:02:45  sparker
// Send only boundary particles
//
//

