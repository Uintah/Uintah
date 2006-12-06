/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation

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

#ifndef Framework_Internal_ApplicationLoader_h
#define Framework_Internal_ApplicationLoader_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Framework/Internal/InternalComponentModel.h>
#include <Framework/Internal/InternalFrameworkServiceInstance.h>

#include <libxml/xmlreader.h>
#include <string>
#include <stack>

#if ! defined(LIBXML_WRITER_ENABLED) && ! defined(LIBXML_OUTPUT_ENABLED)
 #error "Writer or output support not compiled in"
#endif

namespace SCIRun {

class SCIRunFramework;

class ComponentInfo {
};

class ConnectionInfo {
};

// fill in with relevant info
class ApplicationInfo {
public:

private:

};

// TODO: Eventually the following functions should be moved to a service.
// TODO: Need to be able to load SCIRun files too.

class ApplicationLoader : public sci::cca::ports::ApplicationLoaderService,
                          public InternalFrameworkServiceInstance {
public:
  virtual ~ApplicationLoader() {}

  // internal service methods

  /** Factory method for creating an instance of a BuilderService class.
      Returns a reference counted pointer to a newly-allocated BuilderService
      port.  The \em framework parameter is a pointer to the relevent framework
      and the \em name parameter will become the unique name for the new port.*/
  static InternalFrameworkServiceInstance *create(SCIRunFramework* framework);

  /** Creates an instance of the component of type \em className.  The
      parameter \em instanceName is the unique name of the newly created
      instance.  Leave \em instanceName empty to have a unique name generated
      by the framework. This method is implemented through a createComponentInstance
      call to the SCIRunFramework. */
  virtual sci::cca::ComponentID::pointer
  createInstance(const std::string& instanceName,
		 const std::string& className,
		 const sci::cca::TypeMap::pointer &properties);

  /** */
  virtual sci::cca::Port::pointer getService(const std::string &) { return sci::cca::Port::pointer(this); }
  // internal service methods

  virtual std::string getFileName() { return fileName; }
  virtual void setFileName(const std::string& fn);

  virtual void loadFile();
  virtual void loadFile(const std::string& filename);

  virtual void saveFile();
  virtual void saveFile(const std::string& filename);

  static const std::string APP_EXT;

private:
  ApplicationLoader(SCIRunFramework* fwk);

  xmlNode* writeComponentNode(const sci::cca::ComponentID::pointer& cid,
                              const sci::cca::TypeMap::pointer& properties,
                              xmlNode** rootNode);
  xmlNode* writeConnectionNode(const sci::cca::ConnectionID::pointer& cid, xmlNode** rootNode);
  void readComponentNode();
  void readConnectionNode();

  std::string fileName;
  std::stack<xmlNodePtr> nodeStack;
  xmlDocPtr xmlDoc;
};

}

#endif
