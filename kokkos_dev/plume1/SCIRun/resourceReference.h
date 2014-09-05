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
 *  resourceReference.h 
 *
 *  Written by:
 *   Kostadin Damevski & Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2003 
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef CCA_PIDL_resourceReference_h
#define CCA_PIDL_resourceReference_h

#include <iostream>
#include <Core/CCA/SSIDL/array.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/sci_sidl.h>

namespace SCIRun {

/**
 * \class resourceReference
 *
 * A reference to a slave resource from the master framework. This
 * reference contains various methods and fields that are used by
 * the master framework to manage the slave resource.
 *
 */
class resourceReference
{
public:
  
  /** Constructor which takes the name of the resource, the number
      of parallel slave frameworks it represents, and the URLs to
      each of the slave frameworks */
  resourceReference(const std::string& slaveFwkName,
                    const ::SSIDL::array1< ::std::string>& URLs);

  /** Destructor */
  virtual ~resourceReference();

  /** Returns a smart pointer to all of the slave framework(s) */
  sci::cca::Loader::pointer getPtrToAll();
  
  /** Returns the total number of parallel slave framework(s) */
  int getSize();
  
  /** Prints this class to a stream. */
  void print(std::ostream& dbg = std::cout);

  /** List all component types on this resource */
  void listAllComponentTypes(::SSIDL::array1<std::string> &typeList);

  /** Shutdown the loader. */
  int shutdown(float timeout);

  /** ? */
  std::string getName();

  /** ? */
  sci::cca::Component::pointer createInstance(const std::string& name,
                                              const std::string& type,
                                              std::vector<int> nodes);

  /** ? */
  sci::cca::Loader::pointer node(int i);
  
private:
  std::string name;
  int size;
  std::vector<URL> URLs;
  sci::cca::Loader::pointer ploader;
};

} // end namespace SCIRun

#endif














