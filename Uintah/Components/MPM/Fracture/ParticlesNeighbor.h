#ifndef __PARTICLESNEIGHBOR_H__
#define __PARTICLESNEIGHBOR_H__

#include <Uintah/Grid/ParticleSet.h>

#include <list>

namespace Uintah {
namespace MPM {

using std::list;

class ParticlesNeighbor : public list<particleIndex> {
public:
private:
};

} //namespace MPM
} //namespace Uintah

#endif //__PARTICLESNEIGHBOR_H__

// $Log$
// Revision 1.1  2000/06/05 21:15:21  tan
// Added class ParticlesNeighbor to handle neighbor particles searching.
//
