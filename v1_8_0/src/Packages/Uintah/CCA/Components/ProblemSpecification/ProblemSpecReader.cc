#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h> // Only used for MPI cerr
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h> // process determination
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <stdio.h>

#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNode.hpp>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

ProblemSpecReader::ProblemSpecReader(const std::string& filename)
    : filename(filename)
{
}

ProblemSpecReader::~ProblemSpecReader()
{
}

ProblemSpecP ProblemSpecReader::readInputFile()
{
  ProblemSpecP prob_spec;
  
  try {
    XMLPlatformUtils::Initialize();

    // Instantiate the DOM parser.
    XercesDOMParser* parser = new XercesDOMParser;
    parser->setDoValidation(false);
    
    SimpleErrorHandler handler;
    parser->setErrorHandler(&handler);
    
    // Parse the input file
    // No exceptions just yet, need to add
    
    parser->parse(filename.c_str());
    
    if(handler.foundError){
      throw ProblemSetupException("Error reading file: "+filename);
    }
    
    // Clone the Node so we can delete the parser but keep the document
    // contents.  THIS NODE WILL NEED TO BE RELEASED MANUALLY LATER!!!!
    DOMNode* node = parser->getDocument()->cloneNode(true);
#if !defined( _AIX )
    DOMDocument* doc = dynamic_cast<DOMDocument*>(node);
#else
    DOMDocument* doc = static_cast<DOMDocument*>(node);
#endif

    if( !doc ) {
      cout << "dynamic_cast to DOMDocument * failed!\n";
      throw InternalError( "dynamic_cast to DOMDocument * failed!\n" );
    }

    delete parser;

    // Add the parser contents to the ProblemSpecP
    prob_spec = scinew ProblemSpec(doc->getDocumentElement());
  } catch(const XMLException& toCatch) {
    char* ch = XMLString::transcode(toCatch.getMessage());
    string ex("XML Exception: " + string(ch));
    delete [] ch;
    throw ProblemSetupException(ex);
  }
  return prob_spec;
}




