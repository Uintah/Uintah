// copyright etc.


#include <Packages/Ptolemy/Core/PtolemyInterface/ptolemy_scirun_SCIRunJNIActor.h>
#include <Packages/Ptolemy/Core/PtolemyInterface/PTIISCIRun.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>

#include <iostream>


JNIEXPORT jint JNICALL
Java_ptolemy_scirun_SCIRunJNIActor_getScirun(JNIEnv *env, jobject obj)
{
    StartSCIRun *start = new StartSCIRun();
    Thread *t = new Thread(start, "start scirun", 0, Thread::NotActivated);
    t->setStackSize(1024*1024);
    t->activate(false);
    t->detach();

    return 0;
}

JNIEXPORT void JNICALL Java_ptolemy_scirun_SCIRunJNIActor_sendRecordToken(JNIEnv *, jobject)
{
std::cerr << "Java_ptolemy_scirun_SCIRunJNIActor_sendRecordToken" << std::endl;
    SCIRunTest *st = new SCIRunTest();
    Thread *tt = new Thread(st, "try method", 0, Thread::NotActivated);
    tt->setStackSize(1024*1024);
    tt->activate(false);
    tt->detach();

    return;
}
