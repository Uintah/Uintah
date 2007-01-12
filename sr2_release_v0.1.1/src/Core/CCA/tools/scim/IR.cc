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

#include "IR.h"
#include <iostream>
using namespace std;


//Main IR class:

IR::IR()
  : forallMap(NULL)
{
}

IR::~IR()
{
  for(unsigned int i=0; i<outPorts.size(); i++) {
    delete outPorts[i];
  }
  for(unsigned int i=0; i<inPorts.size(); i++) {
    delete inPorts[i];
  }
  for(unsigned int i=0; i<inoutPorts.size(); i++) {
    delete inoutPorts[i];
  }
}

void IR::outFile(std::string filename)
{
  std::ofstream ofile;
  ofile.open(filename.c_str());
  ofile << "OUTPORTS: *************************\n";
  for(unsigned int i=0; i<outPorts.size(); i++) {
    ofile << outPorts[i]->out();
  }
  ofile << "INPORTS: *************************\n";
  for(unsigned int i=0; i<inPorts.size(); i++) {
    ofile << inPorts[i]->out();
  }
  ofile << "INOUTPORTS: *************************\n";
  for(unsigned int i=0; i<inoutPorts.size(); i++) {
    ofile << inoutPorts[i]->out();
  }
  ofile << "MAPS: *************************\n";
  for(unsigned int i=0; i<maps.size(); i++) {
    ofile << maps[i]->out();
  }
  ofile.close();
}

void IR::setPackage(IrPackage* irPkg) { 
  ptrPkg = irPkg;
}

IrPackage* IR::getPackage() {
  return ptrPkg;
}

void IR::addInDefList(IrDefList* inList) {
  inList->modeTag(MODEIN);
  inPorts.push_back(inList);
}

void IR::addOutDefList(IrDefList* outList) {
  outList->modeTag(MODEOUT);
  outPorts.push_back(outList);
}

void IR::addInOutDef(IrDefinition* def) {
  def->mode = MODEINOUT;
  inoutPorts.push_back(def);
}

void IR::addMap(IrMap* map) {
  maps.push_back(map);
}

void IR::setForAllMap(IrMap* map) {
  forallMap = map;
}

IrMap* IR::getForAllMap() {
  return forallMap;
}

int IR::getMapSize() {
  return (int)(maps.size());
}

IrMap* IR::getMap(int i) {
  if(i < maps.size())
    return maps[i];
  else
    return NULL;
}

void IR::omitInOut(std::string portname) {
  if(inoutOmit.find(portname) == inoutOmit.end())
    inoutOmit.insert(portname);
}

void IR::remapInOut(std::string portname) {
  std::set<std::string, ltstr>::iterator iter;
  iter = inoutOmit.find(portname);
  if(iter != inoutOmit.end()) {
    inoutOmit.erase(iter);
  }
}

void IR::omitInOut() {
  inoutOmit.clear();
  for(unsigned int i=0; i<inoutPorts.size(); i++) {
    std::vector<std::string > vecstr = inoutPorts[i]->getPortNames();
    inoutOmit.insert(vecstr.begin(),vecstr.end());
  }
}

int IR::existsOmit(std::string portname) {
  return (inoutOmit.find(portname) != inoutOmit.end());
}

/////////////////////////////////////
//Out methods (for debugging):

std::string IrArgument::out() 
{ 
  return "      <n:argument mode=\"" + mode + "\" type=\"" + type + "\" name=\"" 
    + name + "\"></n:argument>\n"; 
} 

std::string IrArgumentList::out() 
{ 
  std::string mOut;
  for(unsigned int i=0;i<irArgs.size();i++) mOut+=irArgs[i]->out();
  return mOut;
} 

std::string IrMethod::out() 
{ 
  return "    <n:method name=\"" + name + "\" retType=\"" + retType + "\">\n" 
    + irArgL->out() + "    </n:method>\n"; 
} 

std::string IrMethodList::out() 
{ 
  std::string mOut;
  for(unsigned int i=0;i<IrMethods.size();i++) mOut+=IrMethods[i]->out();
  return mOut;
} 

std::string IrPort::out() 
{ 
  return "  <n:port name=\"" + name + "\">\n" + IrMethodL->out() + "  </n:port>\n"; 
} 

std::string IrDefList::out() 
{ 
  std::string portOut;
  for(unsigned int i=0;i<irDefs.size();i++) portOut+=irDefs[i]->out();
  return portOut;
} 

std::string IrPackage::out() 
{ 
  return "<n:package name=\"" + name + "\">\n" + IrPortL->out() + "</n:package>\n"; 
}

std::string IrMap::out()
{
  std::string mapOut;
  mapOut = "map " + inSymbol + " --> " + outSymbol + "\n";
  for(unsigned int i=0;i<methodMaps.size();i++) mapOut+=methodMaps[i]->out(); 
  return mapOut;
}

std::string IrMethodMap::out()
{
  return "\tmethod map " + inMethod->name + " --> " + outMethod->name + "\n";
}

////////////

//******************
//IrDefinition:
IrDefinition::IrDefinition() {}
IrDefinition::~IrDefinition() {}

//****************
//IrArgument:
IrArgument::IrArgument(std::string a_mode, std::string a_type, std::string a_name)
  : mode(a_mode), type(a_type), name(a_name) {}

IrArgument::~IrArgument() {}

std::string IrArgument::getMode() {
  return mode;
}

std::string IrArgument::getType() {
  return type;
}

std::string IrArgument::getName() {
  return name;
}

std::string IrArgument::getMappedType() {
  return mappedType;
}

//****************
//IrArgumentList
IrArgumentList::IrArgumentList() {}

IrArgumentList::~IrArgumentList() {
  for(unsigned int i=0; i<irArgs.size(); i++) delete irArgs[i];
}

void IrArgumentList::addArg(IrArgument* irArg) { 
  irArgs.push_back(irArg);
}

void IrArgumentList::addArgToFront(IrArgument* irArg) {
  irArgs.insert(irArgs.begin(), irArg);
}

int IrArgumentList::getArgSize() {
  return (int)irArgs.size();
}

IrArgument* IrArgumentList::getArg(int i) {
  if((i > -1)&&((int)irArgs.size() > i)) 
    return irArgs[i];
  else
    return NULL;
}

//****************
//IrMethod
IrMethod::IrMethod(std::string a_name, std::string a_retType, IrArgumentList* a_irArgL)
  : name(a_name), retType(a_retType), irArgL(a_irArgL) { }

IrMethod::~IrMethod() {
  delete irArgL;
}

std::string IrMethod::getName() {
  return name;
}

std::string IrMethod::getReturnType() {
  return retType;
}

IrArgumentList* IrMethod::getArgList() {
  return irArgL;
}

//*****************
//IrMethodList
IrMethodList::IrMethodList() {}

IrMethodList::~IrMethodList() {
  for(unsigned int i=0; i<IrMethods.size(); i++)  delete IrMethods[i];
}

void IrMethodList::addMethod(IrMethod* IrMethod) 
{
  IrMethods.push_back(IrMethod);
}


//*******************
//IrPort
IrPort::IrPort(std::string a_name, IrMethodList* a_IrMethodL)
  : name(a_name), IrMethodL(a_IrMethodL) { }

IrPort::~IrPort() { 
  if(package) delete package;
  delete IrMethodL;
}

std::string IrPort::getPackageName() {
  if(package) {
    return package->getName();
  } else {
    return "";
  }
}

std::vector<std::string > IrPort::getPortNames() {
  std::vector<std::string > vecstr;
  vecstr.push_back(name);
  return vecstr;
}

//******************
//IrDefList
IrDefList::IrDefList() {}

IrDefList::~IrDefList() {
  for(unsigned int i=0; i<irDefs.size(); i++)  delete irDefs[i];
}

void IrDefList::addDef(IrDefinition* irDef) {
  irDefs.push_back(irDef);
}

void IrDefList::modeTag(mode_T mode) {
  for(unsigned int i=0; i<irDefs.size(); i++)  irDefs[i]->mode = mode;
}

std::vector<std::string > IrDefList::getPortNames() {
  std::vector<std::string > vecstr;
  for(unsigned int i=0; i<irDefs.size(); i++) {
    std::vector<std::string > tempvec = irDefs[i]->getPortNames();
    vecstr.insert(vecstr.begin(),tempvec.begin(),tempvec.end());  
  }
  return vecstr;
}

int IrDefList::getSize() {
  return (int)irDefs.size();
}
                                                                                                                                              
IrDefinition* IrDefList::getDef(int i) {
  if((i > -1)&&((int)irDefs.size() > i))
    return irDefs[i];
  else
    return NULL;
}

//*******************
//IrPackage
IrPackage::IrPackage(std::string a_name, IrDefList* a_IrPortL)
  : name(a_name), IrPortL(a_IrPortL) { }

IrPackage::~IrPackage() { 
  delete IrPortL;
}

std::string IrPackage::getName() {
  return name;
}

std::vector<std::string > IrPackage::getPortNames() {
  return IrPortL->getPortNames();
} 

//Command IR:

//***************
//IrMap:
IrMap::IrMap() {}

IrMap::IrMap(std::string a_inSymbol, std::string a_outSymbol, IrNameMapList* a_nameMapL, IR* a_irptr) 
  : inSymbol(a_inSymbol), outSymbol(a_outSymbol), nameMapL(a_nameMapL), irptr(a_irptr) { 
  
  nameMapL->parentmap = this;
}

IrMap::IrMap(IrNameMapList* a_nameMapL, IR* a_irptr)
  : inSymbol("_ALL_"), outSymbol("_ALL_"), nameMapL(a_nameMapL), irptr(a_irptr) { 

  nameMapL->parentmap = this;
}

IrMap::~IrMap() {}

void IrMap::addMethodMap(IrMethodMap* mmap) {
  methodMaps.push_back(mmap);
}

int IrMap::getMethodMapSize() {
  return (int)(methodMaps.size());
}

IrMethodMap* IrMap::getMethodMap(int i) {
  return methodMaps[i];
}

std::string IrMap::getInSymbol() {
  return inSymbol;
}

std::string IrMap::getOutSymbol() {
  return outSymbol;
}
                                                                                                                                                 
std::string IrMap::getInPackage() {
  return inPort->getPackageName(); 
}

std::string IrMap::getOutPackage() {
  return outPort->getPackageName();
}

//IrNameMapList
IrNameMapList::IrNameMapList() {}
IrNameMapList::~IrNameMapList() {}

void IrNameMapList::addNameMap(IrNameMap* irNM) {
  nameMaps.push_back(irNM);
}

//IrNameMap
IrNameMap::IrNameMap(std::string nameOne, std::string nameTwo) 
  : nameOne(nameOne), nameTwo(nameTwo), subList(NULL) { }

IrNameMap::~IrNameMap() {
  if(subList != NULL) delete subList;
}

void IrNameMap::addSubList(IrNameMapList* irNML) {
  subList = irNML;
}


//IrMethodMap:
IrMethodMap::IrMethodMap() {}

IrMethodMap::IrMethodMap(IrMethod* a_inMethod, IrMethod* a_outMethod)
  : inMethod(a_inMethod), outMethod(a_outMethod) { }

IrMethodMap::~IrMethodMap() {}

//////////
//Error Reporting function
void errHandler(std::string errline) {
  std::cout << "FATAL ERROR: " << errline << "\n";
  exit(1);
}
