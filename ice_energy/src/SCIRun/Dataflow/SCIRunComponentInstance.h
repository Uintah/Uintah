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
 *  SCIRunComponentInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_SCIRunComponentInstance_h
#define SCIRun_Framework_SCIRunComponentInstance_h

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <string>
#include <vector>

namespace SCIRun {
class CCAPortInstance;
class Module;


/**
 * \class SCIRunComponentInstance
 *
 * A handle for an instantiation of a SCIRun dataflow component in the SCIRun
 * framework. This class is a container for information about a SCIRun dataflow
 * component instantiation that is used by the framework.
 */
class SCIRunComponentInstance : public ComponentInstance {
public:
  SCIRunComponentInstance(SCIRunFramework* fwk,
                          const std::string& instanceName,
                          const std::string& className,
                          const sci::cca::TypeMap::pointer& tm,
                          Module* module);
  virtual ~SCIRunComponentInstance();

  /** Returns a uses or provides port instance named \em name.  If no such port
      exists, returns a null pointer.*/
  virtual PortInstance* getPortInstance(const std::string& name);

  /** Returns an iterator for the list of ports associated with this component. */
  virtual PortInstanceIterator* getPorts();

  /** Return the stored pointer to the SCIRun Module class. */
  Module* getModule() const { return module; }

private:
  class Iterator : public PortInstanceIterator {
  public:
    Iterator(SCIRunComponentInstance*);
    virtual ~Iterator();
    virtual PortInstance* get();
    virtual bool done();
    virtual inline void next() { idx++; }
  private:
    Iterator(const Iterator&);
    Iterator& operator=(const Iterator&);

    SCIRunComponentInstance* component;
    int idx;
  };
  Module* module;
  std::vector<CCAPortInstance*> specialPorts;
  SCIRunComponentInstance(const SCIRunComponentInstance&);
  SCIRunComponentInstance& operator=(const SCIRunComponentInstance);
};

} // end namespace SCIRun

#endif
