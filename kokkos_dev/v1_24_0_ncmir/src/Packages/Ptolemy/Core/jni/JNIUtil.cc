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


#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ThreadGroup.h>

#include <sci_defs/ptolemy_defs.h>

#include <iostream>

#include "JNIUtil.h"

#if defined (_WIN32)
#define PATH_SEPARATOR ';'
#else
#define PATH_SEPARATOR ':'
#endif

#define OPTION_COUNT 7

namespace Ptolemy {

static SCIRun::Mutex jlock("Java VM lock");
static SCIRun::Semaphore *startup;
static JavaVM *jvm;

JVMThread::JVMThread()
{
    std::cerr << "JVMThread::JVMThread" << std::endl;
}

JVMThread::~JVMThread()
{
    std::cerr << "JVMThread::~JVMThread" << std::endl;
}

void JVMThread::run()
{
    std::cerr << "** JVMThread::run() **" << std::endl;
    JNIUtil::createVM() != JNI_OK;
    startup->up();
}


VergilThread::VergilThread(const std::string& cp, const std::string& mp) :
	configPath(&cp), modelPath(&mp)
{
    std::cerr << "VergilThread::VergilThread" << std::endl;
}

VergilThread::~VergilThread()
{
    std::cerr << "VergilThread::~VergilThread" << std::endl;
}

void VergilThread::run()
{
    std::cerr << "** VergilThread::run() **" << std::endl;
    if (! JNIUtil::vergilApplication(*configPath, *modelPath)) {
	std::cerr << "Error running VergilApplication!" << std::endl;
    }
}


void JNIUtil::getVergilApplication(const std::string& cp, const std::string& mp)
{
    SCIRun::Thread *t = new SCIRun::Thread(new VergilThread(cp, mp), "Ptolemy Thread",
					    0, SCIRun::Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    t->detach();
}


JavaVM* JNIUtil::getJavaVM()
{
    // lock here?
    if (0 == jvm) {
	startup = new SCIRun::Semaphore("JVM startup wait", 0);
	SCIRun::Thread *t = new SCIRun::Thread(new JVMThread(), "JVM Thread",
						0, SCIRun::Thread::NotActivated);
	t->setStackSize(1024*1024);
	t->activate(false);
	t->detach();
	startup->down();
    }
    return jvm;
}

int JNIUtil::createVM()
{
    jint status = JNIUtil::DEFAULT;
    std::cerr << "JNIUtil::createVM" << std::endl;

    jlock.lock();

    JNIEnv *env;
    JavaVMInitArgs vm_args;

    // need -Djava.library.path?
    std::string classpath("-Djava.class.path=");
    classpath.append(PTOLEMY_CLASSPATH);

    char *path = new char[classpath.length() + 1];
    classpath.copy(path, std::string::npos);
    path[classpath.length()] = 0;

    JavaVMOption options[OPTION_COUNT];
    options[0].optionString = path;

    std::string ptIIDir("-Dptolemy.ptII.dir=");
    ptIIDir.append(PTOLEMY_PATH);

    char *dir = new char[ptIIDir.length() + 1];
    ptIIDir.copy(dir, std::string::npos);
    dir[ptIIDir.length()] = 0;
    options[1].optionString = dir;
    // ptinvoke command line parameter
    options[2].optionString = "-Xmx256M";

    // verbose output for debugging purposes
    options[3].optionString = "-verbose:jni";
    options[4].optionString = "-verbose:gc";
    options[5].optionString = "-Xcheck:jni";
    options[6].optionString = "-Xdebug";
//     options[7].optionString = "-Xss10m";
//     options[8].optionString = "-Xms32m";

    memset(&vm_args, 0, sizeof(vm_args));
    vm_args.version = JNI_VERSION_1_4;
    vm_args.nOptions = OPTION_COUNT;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = JNI_FALSE;

    status = JNI_CreateJavaVM(&jvm, (void**) &env, &vm_args);
    if (status != JNI_OK) {
	std::cerr << "Error creating JavaVM: status=" << status << std::endl;
	// if a JVM can't be created, there may already be a JVM instance
	// created in this process
	jsize nVMs, size = 1;

	status = JNI_GetCreatedJavaVMs(&jvm, size, &nVMs);
	if (status == JNI_OK && nVMs > 0) {
	    jvm->AttachCurrentThread((void**) &env, NULL);
	} else {
	    std::cerr << "Error getting JavaVM: status=" << status << std::endl;
	}
    } else {
	// free all local references belonging to the current thread
	jvm->DetachCurrentThread();
    }
    delete[] dir;
    delete[] path;

    jlock.unlock();
    return status;
}

int JNIUtil::destroyJavaVM()
{
    std::cerr << "JNIUtil::destroyJavaVM" << std::endl;
    jint status = JNIUtil::DEFAULT;

    //jlock.lock();
    //status = jvm->DestroyJavaVM();
    //jlock.unlock();
    return status;
}

bool JNIUtil::vergilApplication(const std::string& configPath, const std::string& modelPath)
{
    jlock.lock();
    if (0 == jvm) {
	return false;
    }

    JNIEnv *env;
    jint status = JNIUtil::DEFAULT;

    if ((status = jvm->AttachCurrentThread((void**) &env, NULL)) != JNI_OK) {
	jvm->DetachCurrentThread();
	return false;
    }

    jclass ptCls, stringClass;
    jobject ptObj;
    jobjectArray localArray, strArray;
    jstring str0, str1;
    jmethodID cid;
    jthrowable exc;

    // JNI uses '/' instead of '.' to resolve package names
    jclass localCls = env->FindClass("ptolemy/vergil/VergilApplication");
    if (localCls == 0) {
	return false;
    }

    // keep global reference that won't be garbage collected
    if ((ptCls = (jclass) env->NewGlobalRef(localCls)) == 0) {
	return false;
    }
    // don't need local reference
    env->DeleteLocalRef(localCls);

    if ((cid = env->GetMethodID(ptCls, "<init>", "(Ljava/lang/String;[Ljava/lang/String;)V")) == 0) {
	return false;
    }

    if ((stringClass = env->FindClass("java/lang/String")) == 0) {
	return false;
    }

    jstring localStr = env->NewStringUTF(modelPath.c_str());
    if (localStr == 0) {
	return false;
    }
    if ((str0 = (jstring) env->NewGlobalRef(localStr)) == 0) {
	return false;
    }
    env->DeleteLocalRef(localStr);

    jstring localStr1 = env->NewStringUTF(configPath.c_str());
    if (localStr1 == 0) {
	return false;
    }
    if ((str1 = (jstring) env->NewGlobalRef(localStr1)) == 0) {
	return false;
    }
    env->DeleteLocalRef(localStr1);

    if ((localArray = env->NewObjectArray(1, stringClass, str0)) == 0) {
	return false;
    }
    if ((strArray = (jobjectArray) env->NewGlobalRef(localArray)) == 0) {
	return false;
    }
    env->DeleteLocalRef(localArray);

    jobject localObj = env->NewObject(ptCls, cid, str1, strArray);
    if (localObj == 0) {
std::cerr << "Error instantiating VergilApplication object!" << std::endl;
	return false;
    }
    if ((ptObj = env->NewGlobalRef(localObj)) == 0) {
	return false;
    }
    env->DeleteLocalRef(localObj);

    exc = env->ExceptionOccurred();
    // from:
    // http://java.sun.com/docs/books/jni/html/exceptions.html#26377
    if (exc) {
	env->ExceptionDescribe();
	env->ExceptionClear();
	//jclass newExcCls;
	//newExcCls = env->FindClass("java/lang/Exception");
	//if (newExcCls == NULL) {
	    // Unable to find the exception class, give up.
	//    return false;
	//}
	//env->ThrowNew(newExcCls, "thrown from C code");
	return false;
    }
    env->DeleteGlobalRef(str0);
    env->DeleteGlobalRef(str1);
    env->DeleteGlobalRef(ptCls);
    env->DeleteGlobalRef(strArray);
    env->DeleteGlobalRef(ptObj);
    jlock.unlock();
    return true;
}

}
