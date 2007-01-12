// copyright etc.

// Try smart pointer to Java global refs
// 1. can we keep Java data
// from being GC'd in Ptolemy w/o causing deadlocks?
// see docs for Get/Release primitive array critical fxns...
//
// 2. do global references depend at all on thread-local
// JNI data?  (JNIEnv* ?) -> if not, get JNIEnv* ptr from
// call to Attach thread?
// 2a. is there a way of checking if thread is already attached ->
// check what AttachCurrentThread returns...

#ifndef Packages_Ptolemy_Core_Datatypes_JNIGlobalRef_h
#define Packages_Ptolemy_Core_Datatypes_JNIGlobalRef_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <jni.h>

using namespace SCIRun;

namespace Ptolemy {

class JNIGlobalRef;

typedef LockingHandle<JNIGlobalRef> JNIGlobalRefHandle;

class JNIGlobalRef {
public:
    JNIGlobalRef(JavaVM **jvm, JNIEnv* env, jobject localRef);
    ~JNIGlobalRef();

    jobject globalRef() const { return gref; }


    // needed by LockingHandle
    Mutex lock;
    int ref_cnt;

private:
    jobject gref;
    JavaVM *cachedJVM;
};

}

#endif
