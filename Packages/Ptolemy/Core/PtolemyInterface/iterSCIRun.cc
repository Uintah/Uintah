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

using namespace SCIRun;

typedef string* stringPtr;

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

JNIEXPORT jint JNICALL 
Java_ptolemy_scirun_IterateSCIRun_runOnFiles
  (JNIEnv *env, jobject obj, jobjectArray input1Names, jint size1, jobjectArray input2Names, jint size2, jint numParams)
{
	stringPtr input1;
	stringPtr input2;
	jstring tempString;
	
	//stuff for a result array that is the strings of the files saved.  This may or may not be necessary?
	/*
	
	jobjectArray result;
	jclass strArrCls = env->FindClass("Ljava/lang/String;");
	if(strArrCls == NULL){
		return NULL;  //return fail value
	}
	result = env->NewObjectArray(size, strArrCls, NULL);
	if (result == NULL){
		return NULL;  //probably out of memory so fail
	}
	
	if(i==size-1)
		env->SetObjectArrayElement(result,0,tempString);
	else
		env->SetObjectArrayElement(result,i+1,tempString);
	
	*/
	
	//for each thing in the input work on it
	input1 = new string[size1];
	for(jint i = 0; i < size1; i++){
		tempString = (jstring)env->GetObjectArrayElement(input1Names, i);
		input1[i] = JNIUtils::GetStringNativeChars(env, tempString);

		//std::cout << "input1: " << input1[i] << std::endl;
	}
	input2 = new string[size2];
	for(jint i = 0; i < size2; i++){
		tempString = (jstring)env->GetObjectArrayElement(input2Names, i);
		input2[i] = JNIUtils::GetStringNativeChars(env, tempString);

		//std::cout << "input2: " << input2[i] << std::endl;
	}
	
	Iterate *iter = new Iterate(input1,size1,input2,size2,numParams);
	Thread *t = new Thread(iter, "iterate_on_inputs", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    t->join();
	
	
	//if the commands succeed.  
	//TODO need a call back type thing here but i am not sure how this will work yet
	if(true)
		return 1;
	else
		return 0;

}
