// copyright...


#ifndef Ptolemy_Core_PtolemyInterface_PTIISCIRun_h
#define Ptolemy_Core_PtolemyInterface_PTIISCIRun_h

#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Core/Thread/Runnable.h>

#include <iostream>

using namespace SCIRun;

class StartSCIRun : public Runnable {
public:
  virtual ~StartSCIRun() { std::cerr << "~StartSCIRun" << std::endl; }
    void run();
};

class PTIIWorker : public Runnable {
public:
//PTIIWorker(const jobject &obj, const jclass &cls, const jmethodID &mid) : obj_(obj), cls_(cls), mid_(mid) {}
    PTIIWorker() {}
    virtual ~PTIIWorker() {}
    void run();

private:
//     jobject obj_;
//     jclass cls_;
//     jmethodID mid_;
};

#endif
