
#include "ProblemSpecReader.h"
#include <Uintah/Exceptions/ProblemSetupException.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace Uintah;

using namespace PSECore::XMLUtil;

ProblemSpecReader::ProblemSpecReader(const std::string& filename)
    : filename(filename)
{

}

ProblemSpecReader::~ProblemSpecReader()
{
}

ProblemSpecP ProblemSpecReader::readInputFile()
{
  
  try {
    XMLPlatformUtils::Initialize();
  }
  catch(const XMLException& toCatch) {
      throw ProblemSetupException("XML Exception: "+toString(toCatch.getMessage()));
  }
  
  ProblemSpecP prob_spec;
  try {
      // Instantiate the DOM parser.
      DOMParser parser;
      parser.setDoValidation(false);

      SimpleErrorHandler handler;
      parser.setErrorHandler(&handler);

      // Parse the input file
      // No exceptions just yet, need to add

      cout << "Parsing " << filename << endl;
      parser.parse(filename.c_str());

      if(handler.foundError)
	  throw ProblemSetupException("Error reading file: "+filename);

      // Add the parser contents to the ProblemSpecP d_doc

      DOM_Document doc = parser.getDocument();
      prob_spec = new ProblemSpec(doc.getDocumentElement());
  } catch(const XMLException& ex) {
      throw ProblemSetupException("XML Exception: "+toString(ex.getMessage()));
  }

  return prob_spec;
}

