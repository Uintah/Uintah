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



/*
 *  symTable.h: Kwai symbol table implementation that also does the xml output
 *
 *  Written by:
 *   Kostadin Damevski
 *   School of Computing
 *   University of Utah
 *   Feb 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#ifndef CCA_tools_kwai_symTable_h
#define CCA_tools_kwai_symTable_h

#include <sgi_stl_warnings_off.h>
#include <vector> 
#include <string>
#include <sgi_stl_warnings_on.h>
#include <fstream>
#include <iostream>

namespace SCIRun {
  class kObject;
  struct kArgument;
  struct kArgumentList;
  struct kMethod;
  struct kMethodList;
  class kPort;
  struct kObjList;
  class kPackage;

  class symTable {
  public:
    symTable();
    virtual ~symTable();
    void addPackage(kPackage* kPkg);
    void flush(std::string filename);
    bool isPort(std::string bclass); 
  private:
    kPackage* currentPkg;
    std::vector<kPackage* > kPkgs;
    std::ofstream ofile;
  };

  class kObject {
  public:
    kObject() {}
    virtual ~kObject() {}
    virtual std::string out() = 0;
  };

  struct kArgument {
    kArgument(std::string a_mode, std::string a_type, std::string a_name)
      : mode(a_mode), type(a_type), name(a_name) { }
    std::string out();

    std::string mode;
    std::string type;
    std::string name;
  };
  
  struct kArgumentList {
    kArgumentList() {}
    ~kArgumentList() {for(unsigned int i; i<kArgs.size(); i++)  delete kArgs[i];}
    void addArg(kArgument* kArg);
    std::string out();
    
    std::vector<kArgument* > kArgs;
  };

  struct kMethod {
    kMethod(std::string a_name, std::string a_retType, kArgumentList* a_kArgL)
      : name(a_name), retType(a_retType), kArgL(a_kArgL) { }
    ~kMethod() {delete kArgL;}
    std::string out();

    std::string name;
    std::string retType;
    kArgumentList* kArgL;
  };
  
  struct kMethodList {
    kMethodList() {}
    ~kMethodList() {for(unsigned int i; i<kMethods.size(); i++)  delete kMethods[i];}
    void addMethod(kMethod* kMethod);
    std::string out();
    
    std::vector<kMethod* > kMethods;
  };

  class kPort : public kObject{
  public:
    kPort(std::string a_name, kMethodList* a_kMethodL)
      : name(a_name), kMethodL(a_kMethodL) { }
    ~kPort() { delete kMethodL;}
    std::string out();
    
    std::string name;
    kMethodList* kMethodL;
  };

  struct kObjList {
    kObjList() {}
    ~kObjList() {for(unsigned int i; i<kObjs.size(); i++)  delete kObjs[i];}
    void addObj(kObject* kObj);
    std::string out();
    
    std::vector<kObject* > kObjs;
  };
  
  class kPackage : public kObject{
  public:
    kPackage(std::string a_name, kObjList* a_kPortL)
      : name(a_name), kPortL(a_kPortL) { }
    ~kPackage() { delete kPortL; }
    std::string out();
    
    std::string name;
    kObjList* kPortL;
  };
  
} // End namespace SCIRun

#endif





