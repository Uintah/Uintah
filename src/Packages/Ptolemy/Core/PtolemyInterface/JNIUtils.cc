// copyright...

#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Core/Util/Assert.h>

#include <iostream>

JavaVM* JNIUtils::cachedJVM = 0;
// static jweak Class_C = 0;

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *jvm, void *reserved)
{
std::cerr << "JNI_OnLoad" << std::endl;
     JNIEnv *env;
     JNIUtils::cachedJVM = jvm;  // cache the JavaVM pointer
printf("jvm=%#x\n", (unsigned int) JNIUtils::cachedJVM);
 
     if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4) != JNI_OK) {
        // check for 1.2?  We probably shouldn't bother with anything
        // older than that.
         return JNI_ERR; // JNI version not supported
     }
//      jclass cls;
//      cls = env->FindClass("ptolemy/scirun/SCIRunJNIActor");
//      if (cls == NULL) {
//          return JNI_ERR;
//      }
//      // Use weak global ref to allow class to be unloaded
//      Class_C = env->NewWeakGlobalRef(cls);
//      if (Class_C == NULL) {
//          return JNI_ERR;
//      }

//     // register native functions
//     JNINativeMethod nm[2];
//     nm[0].name = "semUp";
//     // method descriptor assigned to signature field
//     nm[0].signature = "()V";
//     nm[0].fnPtr = (void*) semUp_impl;

//     nm[1].name = "semDown";
//     // method descriptor assigned to signature field
//     nm[1].signature = "()V";
//     nm[1].fnPtr = (void*) semDown_impl;

//     env->RegisterNatives(cls, (const JNINativeMethod*) &nm, 2);

#if 0
//      /* Compute and cache the method ID */
//      MID_C_g = (*env)->GetMethodID(env, cls, "g", "()V");
//      if (MID_C_g == NULL) {
//          return JNI_ERR;
//      }
#endif
     return JNI_VERSION_1_4;
}

#if 0
// JNIEXPORT void JNICALL
// JNI_OnUnload(JavaVM *jvm, void *reserved)
// {
// std::cerr << "JNI_OnUnload" << std::endl;
//      JNIEnv *env;
//      if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4 != JNI_OK)) {
//          return;
//      }
//      env->DeleteWeakGlobalRef(Class_C);
//      return;
// }
#endif

void
JNIUtils::setSCIRunStarted(JNIEnv *env, jobject obj)
{
    jclass cls = env->GetObjectClass(obj);
    ASSERT(cls);

    jmethodID mid = env->GetStaticMethodID(cls, "setSCIRunStarted", "()V");
    ASSERT(mid);
    env->CallStaticVoidMethod(cls, mid);
}

void
JNIUtils::ThrowByName(JNIEnv *env, const char *name, const char *msg)
{
    jclass cls = env->FindClass(name);
    /* if cls is NULL, an exception has already been thrown */
    if (cls != NULL) {
        env->ThrowNew(cls, msg);
    }
    /* free the local ref */
    env->DeleteLocalRef(cls);
}

//char* JNIUtils::GetStringNativeChars(JNIEnv *env, jstring jstr)
std::string
JNIUtils::GetStringNativeChars(JNIEnv *env, jstring jstr)
{
    jclass cls;
    jbyteArray bytes = 0;
    jthrowable exc;
    char *buffer = 0;

    if (env->EnsureLocalCapacity(2) < 0) {
        std::cerr << "GetStringNativeChars: could not ensure local capacity."
                  << std::endl;
        return std::string(); // out of memory error
    }

    cls = env->FindClass("java/lang/String");
    if (cls == NULL) {
        std::cerr << "GetStringNativeChars: could not find java/lang/String"
                  << std::endl;
        return std::string();
    }
    // add java.lang.String arg naming the charset
    jmethodID MID_String_getBytes = env->GetMethodID(cls, "getBytes", "()[B");
    if (MID_String_getBytes == NULL) {
        std::cerr << "GetStringNativeChars: could not get method id for String.getBytes"
                  << std::endl;
        return std::string();
    }
    bytes = (jbyteArray) env->CallObjectMethod(jstr, MID_String_getBytes);
    exc = env->ExceptionOccurred();
    if (!exc) {
        jint len = env->GetArrayLength(bytes);
        buffer = new char[len + 1];
        if (buffer == 0) {
            ThrowByName(env, "java/lang/OutOfMemoryError", 0);
            env->DeleteLocalRef(bytes);
            return std::string();
        }
        env->GetByteArrayRegion(bytes, 0, len, (jbyte *)buffer);
        buffer[len] = 0; // NULL-terminate
    } else {
        env->DeleteLocalRef(exc);
    }
    std::string result(buffer);

    env->DeleteLocalRef(bytes);
    delete[] buffer;

    return result;
}

// get instance method id
// 
// jmethodID JNIUtils::getMethodID(JNIEnv *env, jobject obj, std::string methodName, std::string methodSig)
// {
// }

JNIEnv* JNIUtils::GetEnv()
{
printf("JNIUtils::GetEnv jvm=%#x\n", (unsigned int) JNIUtils::cachedJVM);
    JNIEnv *env;

    if (JNIUtils::cachedJVM->GetEnv((void **)&env, JNI_VERSION_1_4) != JNI_OK) {
        if (JNIUtils::cachedJVM->GetEnv((void **) &env, JNI_VERSION_1_2) != JNI_OK) {
            // throw exception?
            // don't bother with java < 1.2
            std::cerr << "JNIUtils::GetEnv: could not get env" << std::endl;
        }
    }
    ASSERT(env); // abort if env is null for now
    return env;
}

JNIEnv* JNIUtils::AttachCurrentThread()
{
printf("JNIUtils::AttachCurrentThread jvm=%#x\n", (unsigned int) JNIUtils::cachedJVM);
    JNIEnv *env;
    if (JNIUtils::cachedJVM->AttachCurrentThread((void**) &env, NULL) != JNI_OK) {
        // report error
        std::cerr << "JNIUtils::AttachCurrentThread: could not attach thread" << std::endl;
        // return null?
    }
    ASSERT(env); // abort if env is null for now
    return env;

}

Semaphore& JNIUtils::sem()
{
    static Semaphore sem_("jni semaphore", 0);
    return sem_;
}

// void JNICALL semUp_impl(JNIEnv *env, jobject obj)
// {
// std::cerr << "void JNICALL semUp_impl" << std::endl;
//     // try to do semaphore stuff from JNI
//     JNIUtils::sem().up();
// }

// void JNICALL semDown_impl(JNIEnv *env, jobject obj)
// {
// std::cerr << "void JNICALL semDown_impl" << std::endl;
//     // try to do semaphore stuff from JNI
//     JNIUtils::sem().down();
// }
