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
 *  TaoPortInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Tao_TaoPortInstance_h
#define SCIRun_Tao_TaoPortInstance_h

#include <SCIRun/PortInstance.h>
#include <Core/CCA/spec/sci_sidl.h>
#include <map>
#include <string>
#include <vector>

namespace SCIRun {

/**
 * \class TaoPortInstance
 *
 */
class TaoPortInstance : public PortInstance
{
public:
  enum PortType { Uses=0, Provides=1 };
  TaoPortInstance(const std::string& portname, const std::string& classname,
                  PortType porttype);
  TaoPortInstance(const std::string& portname, const std::string& classname,
                  const sci::cca::Port::pointer& port,
                  PortType porttype);
  ~TaoPortInstance();
  /** */
  virtual bool connect(PortInstance*);
  /** */
  virtual PortInstance::PortType portType();
  /** */
  virtual std::string getType();
  /** */
  virtual std::string getModel();
  /** */
  virtual std::string getUniqueName();
  /** */
  virtual bool disconnect(PortInstance*);
  /** */
  virtual bool canConnectTo(PortInstance*);
  /** */
  virtual bool available();
  /** */
  virtual PortInstance *getPeer();
  /** */
  std::string getName();
  /** */
  void incrementUseCount();
  /** */
  bool decrementUseCount();
private:
  friend class TaoComponentInstance;
  friend class BridgeComponentInstance;
  std::string name;
  std::string type;
  sci::cca::TypeMap::pointer properties;
  std::vector<PortInstance*> connections;
  sci::cca::Port::pointer port;
  PortType porttype;
  int useCount;
  
  TaoPortInstance(const TaoPortInstance&);
  TaoPortInstance& operator=(const TaoPortInstance&);
};

} // end namespace SCIRun

#endif
