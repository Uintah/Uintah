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
    fprintf(stderr, "JNIGlobalRef::JNIGlobalRef(): jobject ptr=%#x, this=%#x\n", (unsigned int) gref, (unsigned int) this);
}

JNIGlobalRef::~JNIGlobalRef()
{
    //JNIEnv *env = JNIUtils::AttachCurrentThread();
    JNIEnv *env;
    if (cachedJVM->AttachCurrentThread((void**) &env, NULL) != JNI_OK) {
        std::cerr << "~JNIGlobalRef: could not attach thread" << std::endl;
    } else {
        fprintf(stderr, "JNIGlobalRef::~JNIGlobalRef(): jobject ptr=%#x, \n", (unsigned int) gref, (unsigned int) this);
        env->DeleteGlobalRef(gref);
    }
}


}
