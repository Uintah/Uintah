// copyright etc.


#include <Packages/Ptolemy/Core/PtolemyInterface/pt_S_scirun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>

#include <iostream>



JNIEXPORT jint JNICALL
Java_ptolemy_scirun_StartSCIRun_getScirun(JNIEnv *env, jobject obj, jstring name, jstring file, jstring reader)
{
	std::string nPath = JNIUtils::GetStringNativeChars(env, name);
	std::string dPath = JNIUtils::GetStringNativeChars(env, file);
	std::string modName = JNIUtils::GetStringNativeChars(env, reader);
	
	StartSCIRun *start = new StartSCIRun(nPath,dPath,modName);;
	
    Thread *t = new Thread(start, "start scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*2048);
    t->activate(false);
    t->join();
	
	return 1;
}

JNIEXPORT jint JNICALL
Java_ptolemy_scirun_StartSCIRun_loadNetwork(JNIEnv *env, jobject obj)
{
    AddNet *add = new AddNet();
    Thread *t = new Thread(add, "add network", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);	
	t->join();
	
    return 1;
}

