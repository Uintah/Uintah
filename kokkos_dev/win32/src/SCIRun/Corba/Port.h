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
 *  Port.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Corba_Port_h
#define SCIRun_Corba_Port_h
#include <string>
#include <vector>
class corbaObject;

namespace SCIRun
{
  namespace corba
    {
      class Component;
      /**
       * \class Port
       *
       * Defines the generic interface for a SCIRun::corba port.  A port is either an
       * input port (see InPort) or an output port (see OutPort).
       *
       */
      class Port{
      public:
	Port();
	virtual ~Port();

	void setComponent(Component *com);
	
	void setName(const std::string &name);
	
	std::string getName();

	void setType(const std::string &name);

	std::string getType();
	
	/** Returns \em true if this port is an input port and \em false otherwise. */
	bool isUses();
	
	/** add a connected port to the list */
	void addConnectedPort(Port *port);
	
	/** del a connected port from the list */
	void delConnectedPort(Port *port);

      protected:
	std::string name;
	std::string type;
	bool is_uses;
	std::vector<Port*> connPortList;
	Component * com;
      };
    }
}

#endif
