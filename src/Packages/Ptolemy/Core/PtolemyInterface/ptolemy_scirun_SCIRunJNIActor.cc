// copyright etc.


#include <Packages/Ptolemy/Core/PtolemyInterface/ptolemy_scirun_SCIRunJNIActor.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/Assert.h>

#include <iostream>



JNIEXPORT jint JNICALL
Java_ptolemy_scirun_SCIRunJNIActor_getScirun(JNIEnv *env, jobject obj)
{
//    std::cerr << "Thread " << Thread::self()->getThreadName() << "Java_ptolemy_scirun_SCIRunJNIActor_getScirun" << std::endl;
    StartSCIRun *start = new StartSCIRun();
    Thread *t = new Thread(start, "start scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    //t->detach();
    t->join();

std::cerr << "Thread joined" << std::endl;
    JNIUtils::setSCIRunStarted(env, obj);
    return 0;
}


JNIEXPORT void JNICALL
Java_ptolemy_scirun_SCIRunJNIActor_scirunGetResults(JNIEnv *env, jobject obj, jobjectArray objArray)
{
    printf("scirunGetResults: java object=%#x\n", (unsigned int) obj);

#if 0
//     PTIIWorker *w = new PTIIWorker();
//     Thread *tt = new Thread(w, "wait", 0, Thread::NotActivated);
//     tt->setStackSize(256*256);
//     tt->activate(false);
//     tt->detach();

//     static jclass cls = 0;
//     static jmethodID mid = 0;
//     // SCIRunJNIActor class ref
//     if (0 == cls) {
//         jclass localRefClass = env->GetObjectClass(obj);
//         ASSERT(localRefClass);
//         cls = (jclass) env->NewGlobalRef(localRefClass);
//         env->DeleteLocalRef(localRefClass);
//         ASSERT(cls);
//     }
//     std::cerr << "Try to get method ID for 'getRecordLabels'" << std::endl;
//     if (0 == mid) {
//         mid = env->GetMethodID(cls, "getRecordLabels", "()[Ljava/lang/String;");
//         ASSERT(mid);
//     }
//     std::cerr << "Have method ID for 'getRecordLabels'" << std::endl;
#endif

    ASSERT(objArray);
    jobjectArray globalObjArray = (jobjectArray) env->NewGlobalRef(objArray);
    ASSERT(globalObjArray);

    jsize size = env->GetArrayLength(globalObjArray);
    std::cerr << "Array size=" << size << std::endl;
    for (jsize j = 0; j < size; j++) {
        jstring str = (jstring) env->GetObjectArrayElement(globalObjArray, j);
        // returns empty string if there's an error
        std::string result = JNIUtils::GetStringNativeChars(env, str);
std::cerr << "array element " << j << ": " << result << std::endl;
    }

    env->DeleteGlobalRef(globalObjArray);
}
