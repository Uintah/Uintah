
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
public:
    PerProcessorContext(int size, int scratchsize);
    ~PerProcessorContext();

    inline char* get(int offset, int size) {
        ASSERT(offset+size <= datasize);
	return data+offset;
    }
    inline char* getscratch(int size) {
        ASSERT(size <= scratchsize);
	return scratch;
    }
    Object* shadow_cache[200];
};

} // end namespace rtrt

#endif
