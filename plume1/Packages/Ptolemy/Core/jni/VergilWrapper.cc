
#include <Packages/Ptolemy/Core/jni/VergilWrapper.h>

#include <iostream>

namespace Ptolemy {

extern JavaVM* jvm;
extern Mutex jlock;

void VergilThread::run()
{
    std::cerr << "** VergilThread::run() **" << std::endl;
    jlock.lock();
    if (0 == jvm) {
        // throw exception here - JVM should have been started
        return;
    }

    JNIEnv *env;
    jint status = JNIUtil::DEFAULT;

    if ((status = jvm->AttachCurrentThread((void**) &env, NULL)) != JNI_OK) {
        // check this!
        jvm->DetachCurrentThread();
        return;
    }

    jclass ptCls, stringClass;
    jobject ptObj;
    jobjectArray localArray, strArray;
    jstring str0, str1;
    jmethodID cid;
    jthrowable exc;

    // JNI uses '/' instead of '.' to resolve package names
    jclass localCls = env->FindClass("ptolemy/vergil/VergilApplication");
    if (localCls == 0) {
        return;
    }

    // keep global reference that won't be garbage collected
    if ((ptCls = (jclass) env->NewGlobalRef(localCls)) == 0) {
        return;
    }
    // don't need local reference
    env->DeleteLocalRef(localCls);

    if ((cid = env->GetMethodID(ptCls, "<init>", "(Ljava/lang/String;[Ljava/lang/String;)V")) == 0) {
        return;
    }

    if ((stringClass = env->FindClass("java/lang/String")) == 0) {
        return;
    }

    jstring localStr = env->NewStringUTF(modelPath.c_str());
    if (localStr == 0) {
        return;
    }
    if ((str0 = (jstring) env->NewGlobalRef(localStr)) == 0) {
        return;
    }
    env->DeleteLocalRef(localStr);

    jstring localStr1 = env->NewStringUTF(configPath.c_str());
    if (localStr1 == 0) {
        return;
    }
    if ((str1 = (jstring) env->NewGlobalRef(localStr1)) == 0) {
        return;
    }
    env->DeleteLocalRef(localStr1);

    if ((localArray = env->NewObjectArray(1, stringClass, str0)) == 0) {
        return;
    }
    if ((strArray = (jobjectArray) env->NewGlobalRef(localArray)) == 0) {
        return;
    }
    env->DeleteLocalRef(localArray);

    jobject localObj = env->NewObject(ptCls, cid, str1, strArray);
    if (localObj == 0) {
std::cerr << "Error instantiating VergilApplication object!" << std::endl;
        return;
    }
    if ((ptObj = env->NewGlobalRef(localObj)) == 0) {
        return;
    }
    env->DeleteLocalRef(localObj);

    exc = env->ExceptionOccurred();
    // from:
    // http://java.sun.com/docs/books/jni/html/exceptions.html#26377
    if (exc) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        //jclass newExcCls;
        //newExcCls = env->FindClass("java/lang/Exception");
        //if (newExcCls == NULL) {
            // Unable to find the exception class, give up.
        //    return false;
        //}
        //env->ThrowNew(newExcCls, "thrown from C code");
        return;
    }
    env->DeleteGlobalRef(str0);
    env->DeleteGlobalRef(str1);
    env->DeleteGlobalRef(ptCls);
    env->DeleteGlobalRef(strArray);
    env->DeleteGlobalRef(ptObj);
    jlock.unlock();
}

}
