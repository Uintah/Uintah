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
