// copyright...


#ifndef Ptolemy_Core_PtolemyInterface_PTIISCIRun_h
#define Ptolemy_Core_PtolemyInterface_PTIISCIRun_h

#include <Packages/Ptolemy/Core/PtolemyInterface/JNIUtils.h>
#include <Packages/Ptolemy/Dataflow/Modules/Converters/PTIIDataToNrrd.h>

#include <Core/Containers/Array2.h>
#include <Core/Thread/Runnable.h>
#include <Dataflow/Network/Module.h>

#include <iostream>
#include <jni.h>

using namespace SCIRun;
using namespace Ptolemy;

typedef string* stringPtr;

class StartSCIRun : public Runnable {
public:
    StartSCIRun(const std::string &netPath, const std::string &data,
                const std::string &module /*, jint run */)
      : netName(netPath), dataPath(data), readerName(module) /*, runNet(run) */ {}
    StartSCIRun() : netName(""), dataPath(""), readerName("") /*, runNet(0) */ {}
    virtual ~StartSCIRun() { std::cerr << "~StartSCIRun" << std::endl; }
    void run();

private:
  std::string netName;
  std::string dataPath;
  std::string readerName;
//   jint runNet;
};

class ExecuteModule : public Runnable {
public:
    ExecuteModule() {}
    virtual ~ExecuteModule() {}
    void run();
};

class Iterate : public Runnable {
public:
    Iterate(stringPtr input1, jint s1, stringPtr input2, jint s2, jint numP, string pPath, string format)
	 : doOnce(input1), size1(s1), iterate(input2), size2(s2), numParams(numP), picPath(pPath), picFormat(format) {}
	virtual ~Iterate();
    void run();
	static Semaphore& iterSem();
	
	static std::string returnValue;  //will hold info about result of this runnable.
	
private:
	static bool iter_callback(void *data);
	
    stringPtr doOnce;
	jint size1;
	stringPtr iterate;
	jint size2;
	jint numParams;
	string picPath;
	string picFormat;
};

// class SignalExecuteReady : public Runnable {
// public:
//     SignalExecuteReady() {}
//     virtual ~SignalExecuteReady() {}
//     void run();

// private:
// };

class QuitSCIRun : public Runnable {
public:
    QuitSCIRun() {}
    virtual ~QuitSCIRun() {}
    void run();
};


#endif
