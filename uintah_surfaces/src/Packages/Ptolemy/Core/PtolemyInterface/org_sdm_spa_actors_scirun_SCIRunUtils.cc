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

#include <Packages/Ptolemy/Core/PtolemyInterface/org_sdm_spa_actors_scirun_SCIRunUtils.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>

JNIEXPORT jint JNICALL
Java_org_sdm_spa_actors_scirun_SCIRunUtils_getScirun(JNIEnv *env, jclass cls, jstring name, jstring file, jstring module, jboolean runNet)
{
	std::string nPath = JNIUtils::GetStringNativeChars(env, name);
	std::string dPath = JNIUtils::GetStringNativeChars(env, file);
	JNIUtils::modName = JNIUtils::GetStringNativeChars(env, module);

	StartSCIRun *start = new StartSCIRun(nPath, dPath, JNIUtils::modName, runNet);

    Thread *t = new Thread(start, "start scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*2048);
    t->activate(false);
    t->join();

	return 0;
}

JNIEXPORT void JNICALL
Java_org_sdm_spa_actors_scirun_SCIRunUtils_exitAllSCIRunThreads(JNIEnv *, jclass)
{
    QuitSCIRun *quit = new QuitSCIRun();

    Thread *t = new Thread(quit, "quit scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    t->detach();
}
