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


#ifndef Ptolemy_Core_jni_JNIUtil_h
#define Ptolemy_Core_jni_JNIUtil_h

#include <jni.h>
#include <string>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>

namespace Ptolemy {

using SCIRun::Mutex;
using SCIRun::Runnable;
using SCIRun::Semaphore;
using SCIRun::Thread;

class JVMThread : public SCIRun::Runnable {
public:
    JVMThread() {}
    virtual ~JVMThread() {}
    virtual void run();
};


class JNIUtil {
public:
    static JavaVM* getJavaVM();
    static int createVM();
    static int destroyJavaVM();

    static const int DEFAULT = 1;

private:
    friend class JVMThread;
    static SCIRun::Semaphore *startup;

    JNIUtil();
    ~JNIUtil();
    JNIUtil(const JNIUtil&);
    JNIUtil& operator=(const JNIUtil&);
};

    extern JavaVM* jvm;
    extern Mutex jlock;

}

#endif
