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
 *  CorbaPortInstance.h:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef Framework_Corba_CorbaPortInstance_h
#define Framework_Corba_CorbaPortInstance_h

#include <Framework/PortInstance.h>
#include <map>
#include <string>
#include <vector>
#include <Framework/Corba/Port.h>

namespace SCIRun {

class SCIRun::corba::Port;
class CorbaComponentInstance;

/**
 * \class CorbaPortInstance
 *
 */
class CorbaPortInstance : public PortInstance {
public:
  CorbaPortInstance(CorbaComponentInstance* ci,
                    SCIRun::corba::Port* port,
                    const sci::cca::TypeMap::pointer& properties,
                    SCIRun::PortInstance::PortType type);
  ~CorbaPortInstance();

  /**? */
  virtual bool connect(PortInstance*);

  /** ? */
  virtual SCIRun::PortInstance::PortType portType();

  /** ? */
  virtual std::string getUniqueName();

  /** ? */
  virtual std::string getModel();

  /** ? */
  virtual bool disconnect(PortInstance*);

  /** ? */
  virtual bool canConnectTo(PortInstance *);

  virtual std::string getType();

  virtual sci::cca::TypeMap::pointer getProperties() { return properties; }
  virtual void setProperties(const sci::cca::TypeMap::pointer& tm);

private:
  friend class BridgeComponentInstance;

  CorbaPortInstance(const CorbaPortInstance&);
  CorbaPortInstance& operator=(const CorbaPortInstance&);
  void setDefaultProperties();

  CorbaComponentInstance* ci;
  SCIRun::corba::Port* port;
  SCIRun::PortInstance::PortType porttype;
  sci::cca::TypeMap::pointer properties;

  int nConnections;
};

} // end namespace SCIRun

#endif
