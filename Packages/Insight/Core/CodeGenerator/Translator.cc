/**************************************
 *
 * Translator.cc
 *
 * Written by:
 *   Darby J Van Uitert
 *   SCI Institute
 *   January 2003
 *************************************/

#include "Translator.h"

XALAN_CPP_NAMESPACE_USE;

///////////////////////////////
// Constructors and Destructors
///////////////////////////////
Translator::Translator()
{
  xsl_ = "";
}

Translator::Translator(string file)
{
  xsl_ = file;
}

Translator::~Translator()
{

}

//////////////////////////////
// Gets/Sets
//////////////////////////////
void Translator::set_xsl_file(string f)
{
  xsl_ = f;
}

string Translator::get_xsl_file()
{
  return xsl_;
}


////////////////////////////
// generate
///////////////////////////
bool Translator::translate(string xml_source, string output, FileFormat format)
{
  if(this->xsl_ == "") {
    cerr << "ERROR! Translator XSL file not set!\n";
    return false;
  }

  int theResult = -1;

  // generate the appropriate file
  ////////////////////////
  try
  {
    // Call the static initializer for Xerces.
    XMLPlatformUtils::Initialize();
    
    // Initialize Xalan.
    XalanTransformer::initialize();
    
    // Create a XalanTransformer.
    XalanTransformer theXalanTransformer;
    
    // Do the transform.
    theResult = theXalanTransformer.transform(xml_source.c_str(), this->xsl_.c_str(), output.c_str());
    
    if(theResult != 0)
    {
      cerr << "Error!: " << theXalanTransformer.getLastError() << endl;
    }
    // Terminate Xalan...
    XalanTransformer::terminate();
    // Terminate Xerces...
    XMLPlatformUtils::Terminate();
    // Clean up the ICU, if it's integrated...
    XalanTransformer::ICUCleanUp();
  }
  catch(...)
  {
    cerr << "An unknown error occurred in creation of module's xml file!" << endl;
  }

  theResult;
  if(theResult == -1) {
    return false;
  }
  else {
    return true;
  }
}

bool Translator::translate(DOMNode* xml_source, string output, FileFormat format)
{
  cerr << "Not implemented yet\n";
  return false;
}
