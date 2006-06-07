#ifndef UINTAH_HOMEBREW_ScrubItem_H
#define UINTAH_HOMEBREW_ScrubItem_H

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Containers/TrivialAllocator.h>

#include <Packages/Uintah/Core/Grid/share.h>

namespace Uintah {

using SCIRun::TrivialAllocator;
class VarLabel;

struct ScrubItem {
  ScrubItem* next;
  const VarLabel* label;
  int matl;
  const Patch* patch;
  int dw;
  size_t hash;
  int count;
  
  ScrubItem(const VarLabel* l, int m, const Patch* p, int dw) :
    label(l), matl(m), patch(p), dw(dw), count(0)
  {
    size_t ptr = (size_t) l;

    hash = ptr ^ (m << 3) ^ (p->getID() << 4) ^ (dw << 2);
  }

  bool operator==(const ScrubItem& d) {
    return label == d.label && matl == d.matl && patch == d.patch && dw == d.dw;
  }

  SCISHARE static TrivialAllocator scrub_alloc;

  void* operator new(size_t)
  {
    return scrub_alloc.alloc();
  }

  void operator delete(void* rp, size_t)
  {	
    scrub_alloc.free(rp);
  }

};

}

#endif
