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

class StartSCIRun : public Runnable {
public:
    StartSCIRun(const std::string &netPath, const std::string &data,
                const std::string &module, jint run)
        : netName(netPath), dataPath(data), readerName(module), runNet(run) {}
    StartSCIRun() : netName(""), dataPath(""), readerName(""), runNet(0) {}
    virtual ~StartSCIRun() { std::cerr << "~StartSCIRun" << std::endl; }
    void run();

private:
  std::string netName;
  std::string dataPath;
  std::string readerName;
  jint runNet;
};

class AddModule : public Runnable {
public:
    AddModule(std::string command) : command(command) {}
    virtual ~AddModule() {}
    void run();

private:
    std::string command;
};

class ChangeFile : public Runnable {
public:
    ChangeFile(std::string file) : file(file) {}
    virtual ~ChangeFile() {}
    void run();

private:
    std::string file;
};

class SignalExecuteReady : public Runnable {
public:
    SignalExecuteReady() {}
    virtual ~SignalExecuteReady() {}
    void run();

private:
};


#endif
