// Domain Manager.h - Provides an interface to manage a domain
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   March 2000
//
//  Copyright (C) 2000 SCI Group
//
//  Module DomainManager
//
//  Description: 
//  Input ports: 
//  Output ports:
//

#ifndef SCI_project_DomainManger_h
#define SCI_project_DomainManger_h 1

#include <stdio.h>

#include <SCICore/Datatypes/Domain.h>
#include <SCICore/Datatypes/Field.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Containers/String.h>

#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/DomainPort.h>
#include <PSECore/Datatypes/FieldWrapperPort.h>
#include <PSECore/Datatypes/FieldPort.h>
#include <SCICore/Util/DebugStream.h>

#include <vector>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Util;

class DomainManager : public Module {
private:
  vector<DomainOPort*> oports;
  vector<FieldWrapperIPort*> iports;
  DomainHandle domain;
  int inportnum;
  int outportnum;
  
  FieldWrapperHandle fwh;
  DebugStream dbg;
  
  void grab(const FieldWrapperHandle&);
  void update_lists();
  
public:
  DomainManager(const clString& id);
  virtual ~DomainManager();
  virtual void execute();
  virtual void connection(ConnectionMode mode, int which_port, int);
  DomainHandle get_domain() { return domain;};
  TCLstring geoms;
  TCLstring attribs;
};


}
}


#endif
