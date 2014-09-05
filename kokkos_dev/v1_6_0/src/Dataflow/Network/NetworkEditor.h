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
 *  NetworkEditor.h: Interface to Network Editor class from project
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_NetworkEditor_h
#define SCI_project_NetworkEditor_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Comm/MessageBase.h>
#include <Core/GuiInterface/GuiCallback.h>
#include <string>

namespace SCIRun {
  using std::string;
  class GuiInterface;
  class MessageBase;
  class Module;
  class Network;
  class OPort;

  class PSECORESHARE NetworkEditor : public GuiCallback {
    Network* net;
    void multisend(OPort*);
    void do_scheduling(Module*);
    int first_schedule;
    int schedule;
    void save_network(const string&);
    GuiInterface* gui;
  public:
    NetworkEditor(Network*, GuiInterface* gui);
    ~NetworkEditor();
  private:
    virtual void tcl_command(GuiArgs&, void*);
    void init_notes();
  };
} // End namespace SCIRun

#endif
