// copyright etc.


#include <Packages/Ptolemy/Core/PtolemyInterface/ptolemy_scirun_SCIRunJNIActor.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/Assert.h>

#include <iostream>

// eventually all implementations should be moved out of this file
// and the JNI native implementations should be merely wrapper functions

JNIEXPORT jint JNICALL
Java_ptolemy_scirun_SCIRunJNIActor_getScirun(JNIEnv *env, jobject obj, jstring name,
                                                jstring file, jstring module, jint runNet)
{
	std::string nPath = JNIUtils::GetStringNativeChars(env, name);
	std::string dPath = JNIUtils::GetStringNativeChars(env, file);
	JNIUtils::modName = JNIUtils::GetStringNativeChars(env, module);

	StartSCIRun *start = new StartSCIRun(nPath, dPath, JNIUtils::modName, runNet);

    Thread *t = new Thread(start, "start scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*2048);
    t->activate(false);
    t->join();

	return 1;
}

JNIEXPORT jint JNICALL
Java_ptolemy_scirun_SCIRunJNIActor_sendSCIRunData(JNIEnv *env, jobject obj, jobject scirunData)
{
    jclass cls = env->GetObjectClass(scirunData);
    jfieldID fidType = env->GetFieldID(cls, "type", "I");
    // error check
    ASSERT(fidType);

    jint type = env->GetIntField(scirunData, fidType); 
    ASSERT(type);

    jfieldID fidData = env->GetFieldID(cls, "data", "Ljava/lang/Object;");
    // error check
    ASSERT(fidData);
    jobject localData = env->GetObjectField(scirunData, fidData); 
    ASSERT(localData);
    jobject data = env->NewGlobalRef(localData);
    ASSERT(data);

    bool ret;
    switch((int) type) {
        case JNIUtils::TYPE_MESH:
            ret = JNIUtils::getSCIRunMesh(env, data);
            break;
        default:
          std::cerr << "Unknown type" << std::endl;
    }

    if (! ret) {
        env->DeleteGlobalRef(data);
        return 0;
    }
	
    env->DeleteGlobalRef(data);
    return 1;
}

