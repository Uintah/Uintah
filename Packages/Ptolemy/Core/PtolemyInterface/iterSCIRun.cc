// copyright etc.


#include <Packages/Ptolemy/Core/PtolemyInterface/iterSCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Module.h>


#include <iostream>



JNIEXPORT jint JNICALL
Java_ptolemy_scirun_IterateSCIRun_getScirun(JNIEnv *env, jobject obj, jstring name, jstring file, jstring reader, jint run)
{
		

	std::string nPath = JNIUtils::GetStringNativeChars(env, name);
	std::string dPath = JNIUtils::GetStringNativeChars(env, file);
	std::string modName = JNIUtils::GetStringNativeChars(env, reader);
	
	StartSCIRun *start = new StartSCIRun(nPath,dPath,modName,run);;
	
    Thread *t = new Thread(start, "start scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*2048);
    t->activate(false);
    t->join();
	
	return 1;
}

JNIEXPORT jobjectArray JNICALL 
Java_ptolemy_scirun_IterateSCIRun_runOnFiles (JNIEnv *env, jobject obj, jobjectArray inputNames, jint size)
{
	jobjectArray result;
	jstring tempString;
	
	jclass strArrCls = env->FindClass("Ljava/lang/String;");
	if(strArrCls == NULL){
		return NULL;  //return fail value
	}
	result = env->NewObjectArray(size, strArrCls, NULL);
	if (result == NULL){
		return NULL;  //probably out of memory so fail
	}
	
	
	//for each thing in the input work on it
	for(jint i = 0; i < size; i++){
		tempString = (jstring)env->GetObjectArrayElement(inputNames, i);
		
		std::string temp = JNIUtils::GetStringNativeChars(env, tempString);
		std::cout << "IN C: " << temp << std::endl;
		
		if(i==size-1)
			env->SetObjectArrayElement(result,0,tempString);
		else
			env->SetObjectArrayElement(result,i+1,tempString);
	}
	
	//get the first string and send it out
	tempString = (jstring)env->GetObjectArrayElement(inputNames, 0);

 	//what was here before
	std::string fileName = JNIUtils::GetStringNativeChars(env, tempString);
	
	ChangeFile *cf = new ChangeFile(fileName);
	Thread *t = new Thread(cf, "change file", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    t->join();
	
	return result;
}
