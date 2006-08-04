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
 *  Services.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Corba_Services_h
#define SCIRun_Corba_Services_h

#include <map>
#include <string>
#include <vector>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <SCIRun/Corba/Component.h>
namespace SCIRun
{
  namespace corba
    {
      /**
       * \class Services
       *
       * 
       *
       */
      class Services: public sci::cca::CorbaServices{

      public:
	Services(Component* com);
	virtual ~Services();
	// int .sci.cca.CorbaServices.registerUsesPort(in string portName, in string portType)
	virtual int registerUsesPort(const ::std::string& portName, const ::std::string& portType);

	// int .sci.cca.CorbaServices.unregisterUsesPort(in string portName)
	virtual int unregisterUsesPort(const ::std::string& portName);
  
	// int .sci.cca.CorbaServices.addProvidesPort(in string portName, in string portType, in string ior)
	virtual int addProvidesPort(const ::std::string& portName, const ::std::string& portType, const ::std::string& ior);
  
	// int .sci.cca.CorbaServices.removeProvidesPort(in string portName)
	virtual int removeProvidesPort(const ::std::string& portName);
  
	// string .sci.cca.CorbaServices.getIOR(in string portName)
	virtual ::std::string getIOR(const ::std::string& portName);

	int check();

	int done();

	bool go();

	void letGo();

	static sci::cca::CorbaServices::pointer getCorbaServices(const std::string & url);
      private:
	Component* com;
	bool wait;
	bool ready2go;
      };

    } // end namespace SCIRun::corba

} // end namespace SCIRun

#endif
