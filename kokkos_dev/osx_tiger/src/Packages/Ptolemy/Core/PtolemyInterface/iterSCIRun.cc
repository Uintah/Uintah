/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/

#include <Packages/Ptolemy/Core/PtolemyInterface/org_sdm_spa_actors_scirun_IterateSCIRun.h>
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

JNIEXPORT jstring JNICALL 
Java_org_sdm_spa_actors_scirun_IterateSCIRun_runOnFiles(JNIEnv *env, jobject obj, jobjectArray input1Names, jint size1, jobjectArray input2Names, jint size2, jint numParams, jstring picPath)
{
	stringPtr input1;
	stringPtr input2;
	jstring tempString;
	
	std::string pPath = JNIUtils::GetStringNativeChars(env, picPath);
	
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
	
	Iterate *iter = new Iterate(input1,size1,input2,size2,numParams,pPath);
	Thread *t = new Thread(iter, "iterate_on_inputs", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    t->join();
	
	std::cout << "return will be: " << Iterate::returnValue << std::endl;
	
	return env->NewStringUTF(Iterate::returnValue.c_str());

}
