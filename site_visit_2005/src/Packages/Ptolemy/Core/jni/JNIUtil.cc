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

SCIRun::Semaphore *JNIUtil::startup = new SCIRun::Semaphore("JVM startup wait", 0);

Mutex jlock("Java VM lock");
JavaVM *jvm = 0;


void JVMThread::run()
{
    std::cerr << "** JVMThread::run() **" << std::endl;
    JNIUtil::createVM() != JNI_OK;
    JNIUtil::startup->up();
}

JavaVM* JNIUtil::getJavaVM()
{
    // lock here?
    if (0 == jvm) {
        Thread *t = new Thread(new JVMThread(), "JVM Thread",
                               0, Thread::NotActivated);
        t->setStackSize(1024*1024);
        t->activate(false);
        t->detach();
	    JNIUtil::startup->down();
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

}
