
#ifndef Uintah_Parallel_UintahParallelPort_h
#define Uintah_Parallel_UintahParallelPort_h

#include <Uintah/Grid/RefCounted.h>

class UintahParallelPort : public RefCounted {
public:
    UintahParallelPort();
    virtual ~UintahParallelPort();
};

#endif
