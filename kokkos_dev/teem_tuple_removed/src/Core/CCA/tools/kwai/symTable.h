/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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





