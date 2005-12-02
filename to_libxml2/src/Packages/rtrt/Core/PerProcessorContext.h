
#ifndef PERPROCESSORCONTEXT_H
#define PERPROCESSORCONTEXT_H 1

#include <Packages/rtrt/Core/Assert.h>

namespace rtrt {

class Object;

class PerProcessorContext {
  char* data;
  char* scratch;
  int datasize;
  int scratchsize;
  // This tells objects and materials that you should do debug type
  // things.  It is initialized to 0.  The idea, is that you can turn
  // this on for a single pixel while running.  Since it is an
  // unsigned char, you could fill it with bit flags in the future if
  // you want.
  unsigned char debug_; 
public:
  PerProcessorContext(int size, int scratchsize);
  ~PerProcessorContext();
  
  inline char* get(int offset, int IFASSERT(size)) {
    ASSERT(offset+size <= datasize);
    return data+offset;
  }
  inline char* getscratch(int IFASSERT(size)) {
    ASSERT(size <= scratchsize);
    return scratch;
  }

  // The idea here is you could add masks in the future to test for
  // specific things.
  inline bool debug(unsigned char mask = 0xFF) { return debug_ & mask; }
  inline void debugOn(unsigned char mask = 0x01) { debug_ |= mask; }
  inline void debugOff(unsigned char mask = 0x01) { debug_ &= ~mask; }
};

} // end namespace rtrt

#endif
