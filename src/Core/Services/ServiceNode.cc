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

/* Core/ServiceNode.cc 
 * 
 * auth: Jeroen Stinstra
 * adapted from: ComponentNode.cc
*/

#include <Core/Services/ServiceNode.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/XMLUtil/StrX.h>
#include <Core/Util/RWS.h>


#include <stdlib.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#pragma set woff 3303
#endif
#include <xercesc/util/TransService.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#pragma reset woff 3303
#endif

namespace SCIRun {

void PrintComponentNode(ServiceNode& n)
{
	std::cout << "Service name: " << n.servicename << std::endl;
	std::cout << "Class name: " << n.classname << std::endl;
	std::cout << "Version: " << n.version << std::endl;

	std::map<std::string,std::string>::const_iterator it;
	for (it = n.parameter.begin();it!=n.parameter.end(); ++it)
	{
		std::cout << "Parameter pair: " << it->first << " = " << it->second << std::endl;
	}
    std::cout << std::endl;
}

void ProcessServiceNode(const DOMNode& d, ServiceNode& n)
{

	// Get the service name:
	// This name needs to be unique and is used when connecting to services
	
    const XMLCh* xs = to_xml_ch_ptr("name");
    DOMNode *name = d.getAttributes()->getNamedItem(xs);
    if (name == 0) 
	{
		std::cout << "ERROR: Service has no name." << std::endl;
    }
	else 
	{
		n.servicename = std::string(to_char_ptr(name->getNodeValue()));
    }

	// Get service class name. This is used to launch and load the service
	// when needed. Only when the service is requested it is launched.

    const XMLCh* classxs = to_xml_ch_ptr("class");
    DOMNode *classname = d.getAttributes()->getNamedItem(classxs);
    if (classname == 0) 
	{
		std::cout << "ERROR: Service " << n.servicename <<" has no class object." << std::endl;
    }
	else 
	{
		n.classname = std::string(to_char_ptr(classname->getNodeValue()));
    }

	// Get service classpackage name. This specifies the location of the dynamic file
    // to run to provide this service. If this one is not supplied it is assumed to
    // in the same directory as the class it self
    
    const XMLCh* classpackagexs = to_xml_ch_ptr("classpackage");
    DOMNode *classpackagename = d.getAttributes()->getNamedItem(classpackagexs);
    if (classpackagename != 0) 
	{
		n.classpackagename = std::string(to_char_ptr(classpackagename->getNodeValue()));
    }

	// Get service version number. This is used to keep track of bugs and changes
	
    const XMLCh* versionxs = to_xml_ch_ptr("version");
    DOMNode *versionname = d.getAttributes()->getNamedItem(versionxs);
    if (versionname == 0) 
	{
		std::cout << "ERROR: Service " << n.servicename <<" has no version number" << std::endl;
    }
	else 
	{
		n.version = std::string(to_char_ptr(versionname->getNodeValue()));
    }

	// Go through the parameter list and load parameters into ServiceNode
	
	for (DOMNode *child = d.getFirstChild(); child!=0; child = child->getNextSibling()) 
	{
		std::string pname = std::string(to_char_ptr(child->getNodeName()));
		if (pname == "#text") continue;
		std::string pvalue = std::string(removeLTWhiteSpace(getSerializedChildren(child)));
		n.parameter[pname] = pvalue;
	}
}


int ReadServiceNodeFromFile(ServiceNode& n, const std::string filename)
{
  // Initialize the XML4C system
  try 
  {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) 
  {
    std::cerr << "Error during initialization! :\n" << StrX(toCatch.getMessage()) << std::endl;
    return -1;
  }

  // Instantiate the DOM parser.
  XercesDOMParser parser;
  parser.setDoValidation(false);
  
  try 
  {
    parser.parse(filename.c_str());
  }  
  catch (const XMLException& toCatch) 
  {
    std::cerr << "Error during parsing: '" << filename << "'\nException message is:  " << xmlto_string(toCatch.getMessage());
    return 0;
  }
  
  DOMDocument *doc = parser.getDocument();
  DOMNodeList *list = doc->getElementsByTagName(to_xml_ch_ptr("service"));
  unsigned long nlist = list->getLength();
  if (nlist == 0) {
    std::cout << "ServiceNode.cc: Error parsing xml file: " << filename << "\n";
    return 0;
  }

  for (unsigned long i = 0;i < nlist; i++) {
    DOMNode* node = list->item(i);
    if (!node) {
      std::cerr << "Error: NULL node at top level component" << std::endl;
      return 0;
    }
    ProcessServiceNode(*node, n);
  }
  return 1;
}

} // End namespace SCIRun


