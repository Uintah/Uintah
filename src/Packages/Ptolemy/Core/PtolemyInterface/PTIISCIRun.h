// copyright...


#ifndef Ptolemy_Core_PtolemyInterface_PTIISCIRun_h
#define Ptolemy_Core_PtolemyInterface_PTIISCIRun_h

#include <Core/Thread/Runnable.h>

using namespace SCIRun;

class StartSCIRun : public Runnable {
public:
    virtual ~StartSCIRun() {}
    void run();
};

class SCIRunTest : public Runnable {
public:
    virtual ~SCIRunTest() {}
    void run();
};

#endif
