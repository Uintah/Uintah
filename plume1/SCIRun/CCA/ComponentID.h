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
 *  ComponentID.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_ComponentID_h
#define SCIRun_Framework_ComponentID_h

#include <Core/CCA/spec/sci_sidl.h>

namespace SCIRun {
class SCIRunFramework;

/**
 * \class ComponentID
 *
 * This class is an implementation of the CCA ComponentID interface for SCIRun.
 * The ComponentID is a reference to a component instance.  It includes the
 * unique component name and a framework-specific serialization.  In this
 * implementation, the ComponentID also maintains a pointer to the containing
 * framework and the type name of the component. */
class ComponentID : public sci::cca::ComponentID
{
public:
  ComponentID(SCIRunFramework* framework, const std::string& name);
  virtual ~ComponentID();

  /** Returns the unique name for the referenced component. */
  virtual std::string getInstanceName();

  /** Returns a SCIRunFramework-specific serialization of the ComponentID */
  virtual std::string getSerialization();

  /** A pointer to the framework in which the referenced component was
      instantiated. */
  SCIRunFramework* framework;
  
  /** The type name of the referenced component. */
  const std::string name;
private:
  ComponentID(const ComponentID&);
  ComponentID& operator=(const ComponentID&);
};
}

#endif
