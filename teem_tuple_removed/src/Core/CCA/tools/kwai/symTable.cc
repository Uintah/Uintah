#include <Core/CCA/tools/kwai/symTable.h>
#include <iostream>
using namespace SCIRun;
using namespace std;

symTable::symTable() 
{
}

symTable::~symTable()
{
  for(unsigned int i=0; i<kPkgs.size(); i++) {
    delete kPkgs[i];
  }
}

void symTable::flush(std::string filename)
{
  ofile.open(filename.c_str());
  for(unsigned int i=0; i<kPkgs.size(); i++) {
    ofile << kPkgs[i]->out();
  }
  ofile.close();
}

void symTable::addPackage(kPackage* kPkg) 
{
  currentPkg = kPkg;
  kPkgs.push_back(kPkg);
}

bool symTable::isPort(std::string bclass)
{
  return ( (bclass.find("gov.cca.Port")!=bclass.size()) ||
	   (bclass.find("sci.cca.Port")!=bclass.size()) );
}

std::string kArgument::out() 
{ 
  return "      <n:argument mode=\"" + mode + "\" type=\"" + type + "\" name=\"" 
    + name + "\"></n:argument>\n"; 
} 

void kArgumentList::addArg(kArgument* kArg) 
{
  kArgs.push_back(kArg);
}

std::string kArgumentList::out() 
{ 
  std::string mOut;
  for(unsigned int i=0;i<kArgs.size();i++) mOut+=kArgs[i]->out();
  return mOut;
} 

std::string kMethod::out() 
{ 
  return "    <n:method name=\"" + name + "\" retType=\"" + retType + "\">\n" 
    + kArgL->out() + "    </n:method>\n"; 
} 

void kMethodList::addMethod(kMethod* kMethod) 
{
  kMethods.push_back(kMethod);
}

std::string kMethodList::out() 
{ 
  std::string mOut;
  for(unsigned int i=0;i<kMethods.size();i++) mOut+=kMethods[i]->out();
  return mOut;
} 

std::string kPort::out() 
{ 
  return "  <n:port name=\"" + name + "\">\n" + kMethodL->out() + "  </n:port>\n"; 
} 

void kObjList::addObj(kObject* kObj) 
{
  kObjs.push_back(kObj);
}

std::string kObjList::out() 
{ 
  std::string portOut;
  for(unsigned int i=0;i<kObjs.size();i++) portOut+=kObjs[i]->out();
  return portOut;
} 

std::string kPackage::out() 
{ 
  return "<n:package name=\"" + name + "\">\n" + kPortL->out() + "</n:package>\n"; 
}
