// copyright etc.

#include <Packages/Ptolemy/Core/PtolemyInterface/org_sdm_spa_actors_scirun_SCIRunJNIActor.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIIData.h>
#include <Packages/Ptolemy/Core/Datatypes/JNIGlobalRef.h>

// #include <Core/Thread/Thread.h>
// #include <Core/Thread/Runnable.h>
#include <Core/Util/Assert.h>

#include <iostream>

using namespace Ptolemy;

// eventually all implementations should be moved out of this file
// and the JNI native implementations should be merely wrapper functions

JNIEXPORT jint JNICALL
Java_org_sdm_spa_actors_scirun_SCIRunJNIActor_sendSCIRunData(JNIEnv *env, jobject obj, jobject scirunData)
{
    // PTIIData *ptIIData;
    // cache cls
    jclass cls = env->GetObjectClass(scirunData);
    //.cache cls field "type"
    jfieldID fidType = env->GetFieldID(cls, "type", "I");
    // error check
    ASSERT(fidType);

    jint type = env->GetIntField(scirunData, fidType); 
    ASSERT(type);

    //.cache cls field "data"
    jfieldID fidData = env->GetFieldID(cls, "data", "Ljava/lang/Object;");
    // error check
    ASSERT(fidData);
    jobject localData = env->GetObjectField(scirunData, fidData); 
    ASSERT(localData);
    JNIGlobalRef ref(&JNIUtils::cachedJVM, env, localData);
    env->DeleteLocalRef(localData);
    //ASSERT(data);

    bool ret = false;
    switch((int) type) {
        case JNIUtils::TYPE_MESH:
            //ptIIData = new PTIIMeshData(env, localData);
            //ptIIData->getData(env);
            ret = JNIUtils::getSCIRunMesh(env, ref.globalRef());
            break;
        default:
          std::cerr << "Unknown type" << std::endl;
    }

    env->DeleteLocalRef(cls);

    if (! ret) {
        return 0;
    }

    return 1;
}
