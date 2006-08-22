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
 *  ComponentModel.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/ComponentModel.h>
#include <SCIRun/SCIRunFramework.h>
#include <Core/OS/Dir.h>
#include <Core/Thread/Guard.h>
#include <Core/XMLUtil/XMLUtil.h>

#include <iostream>

#ifndef DEBUG
 #define DEBUG 0
#endif

namespace SCIRun {

static Mutex parserLock("parser lock");

ComponentModel::ComponentModel(const std::string& prefixName, SCIRunFramework* framework)
  : prefixName(prefixName), framework(framework)
{
}

ComponentModel::~ComponentModel()
{
}

bool ComponentModel::haveComponent(const std::string& type)
{
  std::cerr << "Error: this component model does not implement haveComponent, name="
            << type << std::endl;
  return false;
}

ComponentInstance*
ComponentModel::createInstance(const std::string& name,
                               const std::string& type,
                               const sci::cca::TypeMap::pointer &tm)
{
  std::cerr << "Error: this component model does not implement createInstance"
            << std::endl;
  return 0;
}

bool ComponentModel::destroyInstance(ComponentInstance* ic)
{
  std::cerr << "Error: this component model does not implement destroyInstance"
            << std::endl;
  return false;
}



///////////////////////////////////////////////////////////////////////////
// convenience functions

bool parseComponentModelXML(const std::string& file, ComponentModel* model)
{
  Guard g(&parserLock);
  static bool initialized = false;

  if (!initialized) {
    // check that libxml version in use is compatible with version
    // the software has been compiled against
    LIBXML_TEST_VERSION;
    initialized = true;
  }

  // create a parser context
  xmlParserCtxtPtr ctxt = xmlNewParserCtxt();
  if (ctxt == 0) {
    std::cerr << "ERROR: Failed to allocate parser context." << std::endl;
    return false;
  }
  // parse the file, activating the DTD validation option
  xmlDocPtr doc = xmlCtxtReadFile(ctxt, file.c_str(), 0, (XML_PARSE_DTDATTR |
                                                          XML_PARSE_DTDVALID |
                                                          XML_PARSE_PEDANTIC));
  // check if parsing suceeded
  if (doc == 0) {
    std::cerr << "ERROR: Failed to parse " << file << std::endl;
    return false;
  }

  // check if validation suceeded
  if (ctxt->valid == 0) {
    std::cerr << "ERROR: Failed to validate " << file << std::endl;
    return false;
  }

  // this code does NOT check for includes!
  xmlNode* node = doc->children;
  for (; node != 0; node = node->next) {
    if (node->type == XML_ELEMENT_NODE &&
        std::string(to_char_ptr(node->name)) == std::string("metacomponentmodel")) {

      xmlAttrPtr nameAttrModel = get_attribute_by_name(node, "name");

      // case-sensitive comparison
      if (std::string(to_char_ptr(nameAttrModel->children->content)) == model->getPrefixName()) {
        xmlNode* libNode = node->children;
        for (;libNode != 0; libNode = libNode->next) {
          if (libNode->type == XML_ELEMENT_NODE &&
              std::string(to_char_ptr(libNode->name)) == std::string("library")) {

            xmlAttrPtr nameAttrLib = get_attribute_by_name(libNode, "name");
            if (nameAttrLib != 0) {
              std::string component_type;
              std::string library_name(to_char_ptr(nameAttrLib->children->content));
#if DEBUG
              std::cerr << "Library name = ->" << library_name << "<-" << std::endl;
#endif
              xmlNode* componentNode = libNode->children;
              for (; componentNode != 0; componentNode = componentNode->next) {
                if (componentNode->type == XML_ELEMENT_NODE &&
                    std::string(to_char_ptr(componentNode->name)) == std::string("component")) {
                  xmlAttrPtr nameAttrComp = get_attribute_by_name(componentNode, "name");
                  if (nameAttrComp != 0) {
                    component_type = std::string(to_char_ptr(nameAttrComp->children->content));
#if DEBUG
                    std::cerr << "Component name = ->" << component_type << "<-" << std::endl;
#endif
                    model->setComponentDescription(component_type, library_name);
                  }
                }
              }
            }
          }
        }
      } else {
        xmlFreeDoc(doc);
        // free up the parser context
        xmlFreeParserCtxt(ctxt);
        xmlCleanupParser();
        return false;
      }
    }
  }
  xmlFreeDoc(doc);
  // free up the parser context
  xmlFreeParserCtxt(ctxt);
  xmlCleanupParser();
  return true;
}

bool
getXMLPaths(SCIRunFramework* fwk, StringVector& xmlPaths)
{
  sci::cca::TypeMap::pointer tm;
  SSIDL::array1<std::string> sArray;

  sci::cca::ports::FrameworkProperties::pointer fwkProperties =
    pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(
                                                             fwk->getFrameworkService("cca.FrameworkProperties", ""));
  if (fwkProperties.isNull()) {
    std::cerr << "Error: Cannot find framework properties" ;
    return false;
  } else {
    tm = fwkProperties->getProperties();
    sArray = tm->getStringArray("sidl_xml_path", sArray);
  }
  fwk->releaseFrameworkService("cca.FrameworkProperties", "");

  for (SSIDL::array1<std::string>::iterator dirIter = sArray.begin();
       dirIter != sArray.end(); dirIter++) {
    StringVector files;
    Dir d(*dirIter);
    d.getFilenamesBySuffix(".xml", files);

    // Dir::getFilenamesBySuffix returns file names only
    for (StringVector::iterator fileIter = files.begin(); fileIter != files.end(); fileIter++) {
      xmlPaths.push_back(*dirIter + "/" + *fileIter);
    }
  }

  return true;
}

StringVector
splitPathString(const std::string& path)
{
  StringVector ans;
  if (path.empty()) {
    return ans;
  }

  // Split the PATH string into a list of paths.  Key on ';' token.
  std::string::size_type start = 0;
  std::string::size_type end = path.find(';', start);
  while (end != path.npos) {
    std::string substring = path.substr(start, end - start);
    ans.push_back(substring);
    start = end + 1;
    end = path.find(';', start);
  }
  // grab the remaining path
  std::string substring = path.substr(start, end - start);
  ans.push_back(substring);

  return ans;
}

} // end namespace SCIRun
