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
 *  TxtBuilder.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#ifndef SCIRun_Framework_TxtBuilder_h
#define SCIRun_Framework_TxtBuilder_h
#include <Core/Thread/Runnable.h>
#include <Core/CCA/spec/cca_sidl.h>

using namespace std;
namespace SCIRun {
  //class BuilderWindow;
  /*
  class myBuilderPort : public virtual sci::cca::ports::BuilderPort {
  public:
    virtual ~myBuilderPort(){}
    virtual void setServices(const sci::cca::Services::pointer& svc);
    virtual void buildRemotePackageMenus(const  sci::cca::ports::ComponentRepository::pointer &reg,
				    const string &frameworkURL);
  protected:
    sci::cca::Services::pointer services;
    //BuilderWindow* builder;
  };
  */
  class TxtBuilder : public sci::cca::Component, public Runnable{
  public:
    TxtBuilder();
    ~TxtBuilder();
    void setServices(const sci::cca::Services::pointer& svc);
  private:
    TxtBuilder(const TxtBuilder&);
    TxtBuilder& operator=(const TxtBuilder&);
    //myBuilderPort builderPort;
    sci::cca::Services::pointer svc;
    sci::cca::ports::BuilderService::pointer bs;
    sci::cca::ports::ComponentRepository::pointer cr;
    void run();
    bool exec_command(char cmdline[]);
    int parse(string cmdline, string args[]);
    void use_cluster(string args[]);
    void list_all(string args[]);
    void list_components(string args[]);
    void list_ports(string args[]);
    void list_compatible(string args[]);
    void list_connections(string args[]);
    void create(string args[]);
    void connect(string args[]);
    void go(string args[]);
    void disconnect(string args[]);
    void destroy(string args[]);
    void execute(string args[]);
    void load(string args[]);
    void save_as(string args[]);
    void save(string args[]);
    void insert(string args[]);
    void clear(string args[]);
    void quit(string args[]);
    void help(string args[]);
    void bad_command(string args[]);
  };
} //namespace SCIRun

#endif
