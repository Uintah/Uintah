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
