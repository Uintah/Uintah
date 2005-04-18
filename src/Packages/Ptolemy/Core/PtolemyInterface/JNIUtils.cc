// copyright...

#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Core/Util/Assert.h>
#include <Dataflow/Network/Network.h>
#include <Core/Thread/Thread.h>

#include <iostream>


// All static variables must be explicitly initialized
// otherwise the Java native library loader will complain.
JavaVM* JNIUtils::cachedJVM = 0;
Network* JNIUtils::cachedNet = 0;
std::string JNIUtils::modName;
// static jweak Class_C = 0;


JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *jvm, void *reserved)
{
     std::cerr << "JNI_OnLoad" << std::endl;
     JNIUtils::cachedJVM = jvm;  // cache the JavaVM pointer
     //printf("jvm=%#x\n", (unsigned int) JNIUtils::cachedJVM);
 
     JNIEnv *env;
     if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4) != JNI_OK) {
         return JNI_ERR; // JNI version not supported
     }
#if 0
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

//      /* Compute and cache the method ID */
//      MID_C_g = (*env)->GetMethodID(env, cls, "g", "()V");
//      if (MID_C_g == NULL) {
//          return JNI_ERR;
//      }
#endif
     return JNI_VERSION_1_4;
}

JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM *jvm, void *reserved)
{
     JNIUtils::cachedJVM = 0;

     //JNIEnv *env;
     //if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4) != JNI_OK) {
     //    return JNI_ERR; // JNI version not supported
     //}
     //env->DeleteWeakGlobalRef(Class_C);

     // detach also called in Module cleanup callback
     jvm->DetachCurrentThread();

     return;
}

bool
JNIUtils::getSCIRunMesh(JNIEnv *env, jobject meshObj)
{
    std::cerr << "JNIUtils::getSCIRunMesh" << std::endl;
    jfieldID fidPts, fidConn;
    jobjectArray pts, conn;

    jclass cls = env->GetObjectClass(meshObj);

    jfieldID fidPtsNum = env->GetFieldID(cls, "connecNum", "I");
    ASSERT(fidPtsNum);
    jint ptsNum = env->GetIntField(meshObj, fidPtsNum);
    jfieldID fidPtsDim = env->GetFieldID(cls, "ptsDim", "I");
    ASSERT(fidPtsDim);
    jint ptsDim = env->GetIntField(meshObj, fidPtsDim);

    jfieldID fidConnNum = env->GetFieldID(cls, "connecNum", "I");
    ASSERT(fidConnNum);
    jint connNum = env->GetIntField(meshObj, fidConnNum);
    jfieldID fidConnDim = env->GetFieldID(cls, "connecDim", "I");
    ASSERT(fidConnDim);
    jint connDim = env->GetIntField(meshObj, fidConnDim);

    jfieldID fidType = env->GetFieldID(cls, "type", "I");
    // error check
    ASSERT(fidType);
    jint type = env->GetIntField(meshObj, fidType);

	Module* mod = cachedNet->get_module_by_id(JNIUtils::modName);
    ASSERT(mod);
//     if (mod == 0) {
//         // hardcode for now - should parse from module name!
//         // WARNING: not supported yet!
//         mod = cachedNet->add_module("Ptolemy", "Converters", "PTIIDataToNrrd");
//     }
    PTIIDataToNrrd* converterMod = dynamic_cast<PTIIDataToNrrd*>(mod);
    ASSERT(converterMod);

    if (type == JNIUtils::TYPE_DOUBLE) {
        converterMod->sendJNIData(ptsNum, connNum, ptsDim, connDim);

        fidPts = env->GetFieldID(cls, "pts", "[[D");
        ASSERT(fidPts);

        // 2D array of doubles: ptsNum x ptsDim
        jobjectArray localPts = (jobjectArray) env->GetObjectField(meshObj, fidPts); 
        ASSERT(localPts);
        pts = (jobjectArray) env->NewGlobalRef((jobject) localPts);
        ASSERT(pts);
        for (int i = 0; i < ptsNum; i++) {
            jdoubleArray dArr = (jdoubleArray) env->GetObjectArrayElement(pts, i);
            ASSERT(dArr);
            // jdouble is double
            double *d = (double *) env->GetPrimitiveArrayCritical(dArr, NULL);
            for (int j = 0; j < ptsDim; j++) {
                converterMod->points(i, j) = d[j];
            }

            // JNI_ABORT - free the buffer without copying back the possible changes
            // in the carray buffer
            // see http://java.sun.com/docs/books/jni/html/functions.html#70415
            // for more details on JNI interface
            env->ReleasePrimitiveArrayCritical(dArr, d, JNI_ABORT);
        }

        fidConn = env->GetFieldID(cls, "connections", "[[D");
        ASSERT(fidConn);

        // 2D array of doubles: connNum x connDim            
        jobjectArray localConn = (jobjectArray) env->GetObjectField(meshObj, fidConn); 
        ASSERT(localConn);
        conn = (jobjectArray) env->NewGlobalRef(localConn);
        ASSERT(conn);
        for (int i = 0; i < connNum; i++) {
            jdoubleArray dArr = (jdoubleArray) env->GetObjectArrayElement(conn, i);
            ASSERT(dArr);
            double *d = (double *) env->GetPrimitiveArrayCritical(dArr, NULL);
            for (int j = 0; j < connDim; j++) {
                converterMod->connections(i, j) = d[j];
            }
            env->ReleasePrimitiveArrayCritical(dArr, d, JNI_ABORT);
        }

        SignalExecuteReady *dataThread = new SignalExecuteReady();
        Thread *t = new Thread(dataThread, "send Ptolemy data", 0, Thread::NotActivated);
        t->setStackSize(1024*1024);
        t->activate(false);
        t->join();


        env->DeleteGlobalRef(pts);
        env->DeleteGlobalRef(conn);
    } else {
        // throw Java exception by name?
        std::cerr << "Unsupported data type" << std::endl;
        return false;
    }
    return true;
}


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
    env->DeleteLocalRef(cls);
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
    //printf("JNIUtils::GetEnv jvm=%#x\n", (unsigned int) JNIUtils::cachedJVM);
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

Semaphore& JNIUtils::dataSem()
{
    static Semaphore sem_("data semaphore", 0);
    return sem_;
}
