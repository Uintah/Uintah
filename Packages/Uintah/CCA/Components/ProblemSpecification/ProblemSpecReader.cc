
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>        // Only used for MPI cerr
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>  // process determination
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <stdio.h>
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

      parser.parse(filename.c_str());

      if(handler.foundError){
	throw ProblemSetupException("Error reading file: "+filename);
      }

      // Add the parser contents to the ProblemSpecP d_doc

      DOM_Document doc = parser.getDocument();
      prob_spec = scinew ProblemSpec(doc.getDocumentElement());
  } catch(const XMLException& ex) {
      throw ProblemSetupException("XML Exception: "+toString(ex.getMessage()));
  }

  return prob_spec;
}

