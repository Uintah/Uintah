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

namespace SCIRun
{
namespace vtk
{
class Port;
class InPort;
class OutPort;

/**
 * \class Component
 *
 * 
 *
 */
class Component{
public:
  Component();
  virtual ~Component();
  
  /** Enable the user interface by setting the \em have_ui to true. The
      argument \em new_thread indicates whether or not the interface is
      rendered in a separate thread(?) */
  void enableUI(bool new_thread=false);
  
  /** Disables the user interface by setting the \em have_ui to false. */
  void disableUI();
  
  /** Returns the value of \em have_ui, which indicates whether or not the
      component defines a user interface. */
  bool haveUI();
  
  /** Returns the port with name \em name */
  Port* getPort(const std::string &name);
  
  /** Add a port to the list of ports available in this component */
  void addPort(Port *port);
  
  /** Remove a port from the list of ports available in this component */
  void removePort(Port *port);
  
  /** Remove a port, using the port name. */
  void removePort(const std::string &name);
  
  /** Returns the number of component ports that are input ports. */
  int numIPorts();
  
  /** Returns the number of component ports that are output ports. */
  int numOPorts(); 
  
  /** Returns \em true if the UI is expected to be threaded and \em false
      otherwise. */
  bool isThreadedUI();
  
  /** Returns the input port indexed by \em index. */
  InPort* getIPort(unsigned int index);
  
  /** Returns the output port indexed by \em index. */
  OutPort* getOPort(unsigned int index);
  
  /** Implements the UI.  Each component type must reimplement this method if
      it provides a user interface. */
  virtual int popupUI();

  void updateAllOutPorts(int flag);
  
private:
  /** A list of all the input ports. */
  std::vector<vtk::InPort*> iports;
  
  /** A list of all the output ports */
  std::vector<vtk::OutPort*> oports;
  
  /** Flag indicating if this component provides a user interface */
  bool have_ui;
  
  /** Flag indicating if this component uses a new thread for its interface */
  bool new_ui_thread;
  
};

} // end namespace SCIRun::vtk

} // end namespace SCIRun

#endif
