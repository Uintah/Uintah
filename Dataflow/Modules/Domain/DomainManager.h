// Domain Manager.h - Provides an interface to manage a domain
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   March 2000
//  Copyright (C) 2000 SCI Group
//  Module DomainManager
//  Description: 
//  Input ports: 
//  Output ports:

#ifndef SCI_project_DomainManger_h
#define SCI_project_DomainManger_h 1

#include <stdio.h>

#include <Core/Datatypes/Domain.h>
#include <Core/Datatypes/Field.h>
#include <Core/TclInterface/TCLvar.h>
#include <Core/Containers/String.h>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/DomainPort.h>
#include <Dataflow/Ports/FieldWrapperPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Util/DebugStream.h>

#include <vector>

namespace SCIRun {


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


} // End namespace SCIRun


#endif
