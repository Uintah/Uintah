// copyright...


#ifndef Ptolemy_Core_PtolemyInterface_PTIISCIRun_h
#define Ptolemy_Core_PtolemyInterface_PTIISCIRun_h

#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Core/Thread/Runnable.h>

#include <iostream>

using namespace SCIRun;

class StartSCIRun : public Runnable {
public:
  StartSCIRun(std::string netPath, std::string data, std::string module) 
  		{ netName = netPath;  dataPath = data;  readerName = module; }
  StartSCIRun() { netName = "";  dataPath = "";  readerName = "";}
  virtual ~StartSCIRun() { netName = "";  std::cerr << "~StartSCIRun" << std::endl; }
  void run();
private:
  std::string netName;
  std::string dataPath;
  std::string readerName;
  
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


class AddNet : public Runnable {
public:
  AddNet();
  virtual ~AddNet() {}
  void run();

  static int pt_flag;  //1 when done, 0 while going
  
private:
    std::string command;
};


#endif
