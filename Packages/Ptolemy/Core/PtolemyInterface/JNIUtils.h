// copyright...


#ifndef Ptolemy_Core_PtolemyInterface_JNIUtils_h
#define Ptolemy_Core_PtolemyInterface_JNIUtils_h

#include <jni.h>

#include <Core/Thread/Semaphore.h>

using SCIRun::Semaphore;

// Meyers Singleton pattern
class JNIUtils {
public:
    static Semaphore& sem();
    // from
    // The Java Native Interface Programmer's Guide and Specification
    // http://java.sun.com/docs/books/jni/html/exceptions.html#11201
    static void JNU_ThrowByName(JNIEnv *env, const char *name, const char *msg);

    // from
    // The Java Native Interface Programmer's Guide and Specification
    // http://java.sun.com/docs/books/jni/html/other.html#29406
    static char* JNU_GetStringNativeChars(JNIEnv *env, jstring jstr);

private:
    JNIUtils();
    JNIUtils(JNIUtils const&);
    JNIUtils& operator=(JNIUtils const&);
    ~JNIUtils();
};



extern JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *jvm, void *reserved);


#endif
