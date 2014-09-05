#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h> // Only used for MPI cerr
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h> // process determination
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

//#include <Core/XMLUtil/SimpleErrorHandler.h>
#include <Core/XMLUtil/XMLUtil.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <stdio.h>

#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMException.hpp> 
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
using namespace std;
using namespace Uintah;
using namespace SCIRun;

ProblemSpecReader::ProblemSpecReader(const std::string& filename)
  : d_filename(filename)
{
}

ProblemSpecReader::~ProblemSpecReader()
{
}

ProblemSpecP
ProblemSpecReader::readInputFile()
{
  if (d_xmlData != 0)
    return d_xmlData;

  ProblemSpecP prob_spec;
  static bool initialized = false;
  try {
    if (!initialized) {
      XMLPlatformUtils::Initialize();
      initialized = true;
    }

    // Instantiate the DOM parser.
    XercesDOMParser* parser = new XercesDOMParser;
    parser->setDoValidation(false);
#if 0    
    ErrorHandler handler;
    parser->setErrorHandler(&handler);
#endif
    
    // Parse the input file
    // No exceptions just yet, need to add
    
    parser->parse(d_filename.c_str());

#if 0    
    if(handler.foundError){
      throw ProblemSetupException("Error reading file: "+d_filename, __FILE__, __LINE__);
    }
#endif
    
    // Adopt the Node so we can delete the parser but keep the document
    // contents.  THIS NODE WILL NEED TO BE RELEASED MANUALLY LATER!!!!
    //DOMNode* node = parser->getDocument()->cloneNode(true);
    DOMDocument* doc = parser->adoptDocument();
    //#if !defined( _AIX )
    //DOMDocument* doc = dynamic_cast<DOMDocument*>(node);
    //#else
    //DOMDocument* doc = static_cast<DOMDocument*>(node);
    //#endif

    if( !doc ) {
      cout << "Parse failed!\n";
      throw InternalError( "Parse failed!\n", __FILE__, __LINE__ );
    }

    delete parser;

    // Add the parser contents to the ProblemSpecP
    prob_spec = scinew ProblemSpec(doc->getDocumentElement());
  } catch(const XMLException& toCatch) {
    char* ch = XMLString::transcode(toCatch.getMessage());
    string ex("XML Exception: " + string(ch));
    delete [] ch;
    throw ProblemSetupException(ex, __FILE__, __LINE__);
  }

  resolveIncludes(prob_spec);
  d_xmlData = prob_spec;
  return prob_spec;
}

void
ProblemSpecReader::resolveIncludes(ProblemSpecP params)
{
  // find the directory the current file was in, and if the includes are 
  // not an absolute path, have them for relative to that directory
  string directory = d_filename;
  int i;
  for (i = directory.length()-1; i >= 0; i--) {
    //strip off characters after last /
    if (directory[i] == '/')
      break;
  }

  directory = directory.substr(0,i+1);

  ProblemSpecP child = params->getFirstChild();
  while (child != 0) {
    if (child->getNodeType() == DOMNode::ELEMENT_NODE) {
      string str = child->getNodeName();
      // look for the include tag
      if (str == "include") {
        map<string, string> attributes;
        child->getAttributes(attributes);
        string href = attributes["href"];

        // not absolute path, append href to directory
        if (href[0] != '/')
          href = directory + href;
        if (href == "")
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        
        // open the file, read it, and replace the index node
        ProblemSpecReader *psr = new ProblemSpecReader(href);
        ProblemSpecP include = psr->readInputFile();
        delete psr;
        // nodes to be substituted must be enclosed in a 
        // "Uintah_Include" node

        if (include->getNodeName() == "Uintah_Include" || 
            include->getNodeName() == "Uintah_specification") {
          ProblemSpecP incChild = include->getFirstChild();
          while (incChild != 0) {
            //make include be created from same document that created params
            ProblemSpecP newnode = child->importNode(incChild, true);
            resolveIncludes(newnode);
            params->getNode()->insertBefore(newnode->getNode(), child->getNode());
            incChild = incChild->getNextSibling();
          }
          ProblemSpecP temp = child->getNextSibling();
          params->removeChild(child);
          child = temp;
          continue;
        }
        else {
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        }
      }
      // recurse on child's children
      resolveIncludes(child);
    }
    child = child->getNextSibling();

  }

}



