/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Component.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_Component_h
#define SCIRun_Vtk_Component_h

#include <SCIRun/PortInstance.h>
#include <map>
#include <string>
#include <vector>
#include <SCIRun/Vtk/InPort.h>
#include <SCIRun/Vtk/OutPort.h>

namespace SCIRun{
  namespace vtk{
    class Port;
    class InPort;
    class OutPort;

    class Component{
    public:
      // default constructor
      Component();

      // virtual destructor
      virtual ~Component();

      // enable UI
      void enableUI(bool new_thread=false);

      // disable UI
      void disableUI();

      // Check if UI is defined
      bool haveUI();

      // Find a port with the given name
      Port* getPort(const std::string &name);

      // add a port
      void addPort(Port *port);

      // remove a port 
      void removePort(Port *port);

      // remove a port by name
      void removePort(const std::string &name);

      // Get number of input ports
      int numIPorts();

      // Get number of output ports
      int numOPorts(); 

      // check if a new thread is requested for UI
      bool isThreadedUI();

      // Get the index-th input port
      InPort* getIPort(unsigned int index);

      // Get the index-th output port
      OutPort* getOPort(unsigned int index);

      // Define the UI action.
      // Component should re-implement this method if UI presents.
      virtual int popupUI();

    private:
      // List of input ports
      std::vector<vtk::InPort*> iports;

      // List of ouput ports
      std::vector<vtk::OutPort*> oports;

      // Flag indicating if having UI
      bool have_ui;

      // Flag indicating if UI uses a new thread
      bool new_ui_thread;
      
    };
  }
}

#endif
