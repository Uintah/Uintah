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
 *  ConnectionID.h: 
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#ifndef SCIRun_Framework_ConnectionID_h
#define SCIRun_Framework_ConnectionID_h

#include <Core/CCA/spec/sci_sidl.h>

namespace SCIRun {

class SCIRunFramework;
class ComponentID;

/**
 * \class ConnectionID
 *
 *  This class is an implementation of the cca::ConnectionID interface, which
 *  describes a connection between CCA components. A connection is made at the
 *  users direction when one component provides a Port that another component
 *  advertises for and uses.  The components are referred to by their opaque
 *  ComponentID references and the Ports are referred to by their string
 *  instance names.
 */
class ConnectionID : public sci::cca::ConnectionID
{
public:
  ConnectionID(const sci::cca::ComponentID::pointer& user,
               const std::string& uPortName,
               const sci::cca::ComponentID::pointer& provider,
               const std::string& pPortName);
  virtual ~ConnectionID();

  /** Returns a smart pointer to the component that is the port provider. */
  virtual sci::cca::ComponentID::pointer getProvider();

  /** Returns a smart pointer to the component that is the port user. */
  virtual sci::cca::ComponentID::pointer getUser();

  /** Returns the provides port name from the providing component. */
  virtual std::string getProviderPortName();

  /** Returns the uses port name from the using component. */
  virtual std::string getUserPortName();

  inline sci::cca::TypeMap::pointer getProperties() { return properties; }
  inline void setProperties(const sci::cca::TypeMap::pointer &tm) { properties = tm; }

private:
  ConnectionID(const ConnectionID&);
  ConnectionID& operator=(const ConnectionID&);
  std::string userPortName;
  std::string providerPortName;
  sci::cca::ComponentID::pointer user;
  sci::cca::ComponentID::pointer provider;
  sci::cca::TypeMap::pointer properties;
};

} // end namespace SCIRun

#endif

