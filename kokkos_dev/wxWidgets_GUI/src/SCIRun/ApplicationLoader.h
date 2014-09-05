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

#ifndef SCIRun_ApplicationLoader_h
#define SCIRun_ApplicationLoader_h

#include <Core/CCA/spec/cca_sidl.h>

#include <libxml/xmlreader.h>
#include <string>
#include <stack>

#if ! defined(LIBXML_WRITER_ENABLED) && ! defined(LIBXML_OUTPUT_ENABLED)
 #error "Writer or output support not compiled in"
#endif

namespace SCIRun {

// TODO: Eventually the following functions should be moved to a service.
// TODO: Need to be able to load SCIRun files too.

class ApplicationLoader {
public:
  ApplicationLoader(const std::string& fn) : fileName(fn) {}
  const std::string& getFileName() const { return fileName; }
  void setFileName(const std::string& fn);

  bool loadNetworkFile();

  bool saveNetworkFileAs(const sci::cca::ports::BuilderService::pointer& bs, const sci::cca::GUIBuilder::pointer& gs, const std::string& fn);
  //bool saveNetworkFileAs(const sci::cca::ports::BuilderService::pointer& bs, const std::string& fn) { return false; }

  bool saveNetworkFile(const sci::cca::ports::BuilderService::pointer& bs, const sci::cca::GUIBuilder::pointer& gs);
  //bool saveNetworkFile(const sci::cca::ports::BuilderService::pointer& bs) { return false; }

  static const std::string APPLICATION_FILE_EXTENSION;

private:
  void writeComponentNode();
  void writeConnectionNode();
  void readComponentNode();
  void readConnectionNode();

  std::string fileName;
  std::stack<xmlNodePtr> nodeStack;
  xmlDocPtr xmlDoc;

};

}

#endif
