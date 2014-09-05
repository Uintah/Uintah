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
 *  SCIRunLoader.h: An instance of the SCIRun Parallel Component Loader
 *
 *  Written by:
 *   Kosta  & Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#ifndef SCIRun_SCIRunLoader_h
#define SCIRun_SCIRunLoader_h

#include <Core/Thread/Mutex.h>
#include <Core/CCA/spec/sci_sidl.h>
#include <vector>
#include <map>
#include <string>
#include <SCIRun/ComponentModel.h>
#include <SCIRun/CCA/CCAComponentDescription.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/resourceReference.h>

namespace SCIRun {

/**
 * \class SCIRunLoader
 *
 *
 *
 */
class SCIRunLoader : public sci::cca::Loader
{
public:
  SCIRunLoader(const std::string& loaderName, const std::string& frameworkURL);
  virtual ~SCIRunLoader();

  /** */
  virtual int createPInstance(const ::std::string& componentType,
                              const std::string& componentName,
                              const sci::cca::TypeMap::pointer& properties,
                              SSIDL::array1<std::string>& componentURLs);

  /** */
  virtual int createInstance(const std::string& componentType,
                             const std::string& componentName, 
                             const sci::cca::TypeMap::pointer& properties,
                             std::string &componentURL);
  
  /** */
  virtual int destroyInstance(const std::string& componentName, float timeout);
  
  /** */
  virtual int getAllComponentTypes(::SSIDL::array1< ::std::string>& componentTypes);

  /** */
  virtual int shutdown(float timeout);

  /** */
  int mpi_rank;
  int mpi_size;
  
private:
  void buildComponentList();
  void readComponentDescription(const std::string& file);
  void destroyComponentList();
  std::string masterFrameworkURL;

  typedef std::map<std::string, CCAComponentDescription*> componentDB_type;
  componentDB_type components;
  SCIRun::Mutex lock_components;
};

} // end namespace SCIRun

#endif

