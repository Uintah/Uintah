// copyright etc.


#include <Packages/Ptolemy/Core/Datatypes/JNIGlobalRef.h>
#include <Core/Util/Assert.h>
#include <iostream>


namespace Ptolemy {

JNIGlobalRef::JNIGlobalRef(JavaVM **jvm, JNIEnv *env, jobject localRef) : lock("JNI reference lock"), ref_cnt(0), cachedJVM(*jvm)
{
    // throw exception here eventually...
    ASSERT(env);
    gref = env->NewGlobalRef(localRef);
    ASSERT(gref);
}

JNIGlobalRef::~JNIGlobalRef()
{
    //JNIEnv *env = JNIUtils::AttachCurrentThread();
    JNIEnv *env;
    if (cachedJVM->AttachCurrentThread((void**) &env, NULL) != JNI_OK) {
        // report error
        std::cerr << "~JNIGlobalRef: could not attach thread" << std::endl;
    }
    env->DeleteGlobalRef(gref);
}


}
