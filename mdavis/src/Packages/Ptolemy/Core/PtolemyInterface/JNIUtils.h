// copyright...


#ifndef Ptolemy_Core_PtolemyInterface_JNIUtils_h
#define Ptolemy_Core_PtolemyInterface_JNIUtils_h

#include <jni.h>
#include <string>

#include <Core/Containers/Array2.h>
#include <Core/Thread/Semaphore.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/Module.h>
#include <Packages/Ptolemy/Dataflow/Modules/Converters/PTIIDataToNrrd.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/Datatypes/JNIGlobalRef.h>

using namespace SCIRun;
using namespace Ptolemy;

class JNIUtils {
public:
    // make sure this matches SCIRunData class
    enum {
        TYPE_MESH = 1,
        TYPE_DOUBLE = 100,
        TYPE_FLOAT,
        TYPE_INT
    } scirunData;

    static Semaphore& sem();
    static Semaphore& dataSem();
    static JavaVM *cachedJVM;
    static Network* cachedNet;
    static std::string modName;

    static JNIGlobalRef* dataObjRef;

    static const int DEFAULT = 1;

    static void setSCIRunStarted(JNIEnv *env, jobject obj);
    static bool getMesh(NrrdDataHandle points_handle_, NrrdDataHandle connections_handle_);

    // from
    // The Java Native Interface Programmer's Guide and Specification
    // http://java.sun.com/docs/books/jni/html/exceptions.html#11201
    static void ThrowByName(JNIEnv *env, const char *name, const char *msg);

    // from
    // The Java Native Interface Programmer's Guide and Specification
    // http://java.sun.com/docs/books/jni/html/other.html#29406
    // returns empty string if there's an error
    static std::string GetStringNativeChars(JNIEnv *env, jstring jstr);

    // from
    // The Java Native Interface Programmer's Guide and Specification
    // http://java.sun.com/docs/books/jni/html/other.html#30439
    //
    // JNIEnv* points to thread-local data - see
    //  http://java.sun.com/docs/books/jni/html/design.html#8371
    static JNIEnv* GetEnv();

    static JNIEnv* AttachCurrentThread();

private:
    JNIUtils();
    JNIUtils(JNIUtils const&);
    JNIUtils& operator=(JNIUtils const&);
    ~JNIUtils();

    const static int UNSTRUCTURED_REGULAR_NRRD_DIM = 2;
    const static int DEFAULT_NUM_POINTS = 100;
    const static int DEFAULT_POINTS_DIM = 3;

};

// void JNICALL semUp_impl(JNIEnv *env, jobject self);
// void JNICALL semDown_impl(JNIEnv *env, jobject self);

extern JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *jvm, void *reserved);


extern JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM *jvm, void *reserved);


#endif
