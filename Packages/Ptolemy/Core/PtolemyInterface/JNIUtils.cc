// copyright...

#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>

#include <iostream>

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *jvm, void *reserved)
{
std::cerr << "JNI_OnLoad" << std::endl;
    //sem_test = scinew Semaphore("test semaphore", 0);

     JNIEnv *env;
     //PTIIJNIUtil::cachedJVM = jvm;  // cache the JavaVM pointer
 
     if (jvm->GetEnv((void **)&env, JNI_VERSION_1_2)) {
         return JNI_ERR; // JNI version not supported
     }
#if 0
//      jclass cls;
//      cls = (*env)->FindClass(env, "C");
//      if (cls == NULL) {
//          return JNI_ERR;
//      }
//      /* Use weak global ref to allow C class to be unloaded */
//      Class_C = (*env)->NewWeakGlobalRef(env, cls);
//      if (Class_C == NULL) {
//          return JNI_ERR;
//      }
//      /* Compute and cache the method ID */
//      MID_C_g = (*env)->GetMethodID(env, cls, "g", "()V");
//      if (MID_C_g == NULL) {
//          return JNI_ERR;
//      }
#endif
     return JNI_VERSION_1_2;
}


void
JNIUtils::JNU_ThrowByName(JNIEnv *env, const char *name, const char *msg)
{
    jclass cls = env->FindClass(name);
    /* if cls is NULL, an exception has already been thrown */
    if (cls != NULL) {
        env->ThrowNew(cls, msg);
    }
    /* free the local ref */
    env->DeleteLocalRef(cls);
}

char*
JNIUtils::JNU_GetStringNativeChars(JNIEnv *env, jstring jstr)
{
    jclass cls;
    jbyteArray bytes = 0;
    jthrowable exc;
    char *result = 0;

    if (env->EnsureLocalCapacity(2) < 0) {
        std::cerr << "JNU_GetStringNativeChars: could not ensure local capacity."
                  << std::endl;
        return 0; /* out of memory error */
    }

    cls = env->FindClass("java/lang/String");
    if (cls == NULL) {
        std::cerr << "JNU_GetStringNativeChars: could not find java/lang/String"
                  << std::endl;
        return result;
    }
    // add java.lang.String arg naming the charset
    jmethodID MID_String_getBytes = env->GetMethodID(cls, "getBytes", "()[B");
    if (MID_String_getBytes == NULL) {
        std::cerr << "JNU_GetStringNativeChars: could not get method id for String.getBytes"
                  << std::endl;
        return result;
    }
    bytes = (jbyteArray) env->CallObjectMethod(jstr, MID_String_getBytes);
    exc = env->ExceptionOccurred();
    if (!exc) {
        jint len = env->GetArrayLength(bytes);
        result = (char *)malloc(len + 1);
        if (result == 0) {
            JNU_ThrowByName(env, "java/lang/OutOfMemoryError", 0);
            env->DeleteLocalRef(bytes);
            return 0;
        }
        env->GetByteArrayRegion(bytes, 0, len, (jbyte *)result);
        result[len] = 0; /* NULL-terminate */
    } else {
        env->DeleteLocalRef(exc);
    }
    env->DeleteLocalRef(bytes);
    return result;
}


Semaphore& JNIUtils::sem()
{
    static Semaphore sem_("jni semaphore", 0);
    return sem_;
}
