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
  cc_xsl_ = "";
  gui_xsl_ = "";
  xml_xsl_ = "";
}

Translator::Translator(string cc, string gui, string xml)
{
  cc_xsl_ = cc;
  gui_xsl_ = gui;
  xml_xsl_ = xml;
}

Translator::~Translator()
{

}

//////////////////////////////
// Gets/Sets
//////////////////////////////
void Translator::set_cc_xsl(string f)
{
  cc_xsl_ = f;
}

string Translator::get_cc_xsl()
{
  return cc_xsl_;
}

void Translator::set_gui_xsl(string f)
{
  gui_xsl_ = f;
}

string Translator::get_gui_xsl()
{
  return gui_xsl_;
}

void Translator::set_xml_xsl(string f)
{
  xml_xsl_ = f;
}

string Translator::set_xml_xsl()
{
  return xml_xsl_;
}

////////////////////////////
// generate
///////////////////////////
bool Translator::translate(string xml_source, string cc_output, string gui_output, string xml_output)
{
  if(this->cc_xsl_ == "") {
    cerr << "ERROR! Translator xsl CC file not set!\n";
    return false;
  }

  if(this->gui_xsl_ == "") {
    cerr << "ERROR! Translator xsl GUI file not set!\n";
    return false;
  }

  if(this->xml_xsl_ == "") {
    cerr << "ERROR! Translator xsl XML file not set!\n";
    return false;
  }

  int theResult = -1;

  // generate the cc file
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
    theResult = theXalanTransformer.transform(xml_source.c_str(), this->cc_xsl_.c_str(), cc_output.c_str());

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
  // generate the gui file
  ////////////////////////
  try
  {
    XMLPlatformUtils::Initialize();
    XalanTransformer::initialize();
    XalanTransformer theXalanTransformer;
    theResult = theXalanTransformer.transform(xml_source.c_str(), this->gui_xsl_.c_str(), gui_output.c_str());
    if(theResult != 0)
    {
      cerr << "Error: " << theXalanTransformer.getLastError() << endl;
    }
    XalanTransformer::terminate();
    XMLPlatformUtils::Terminate();
    XalanTransformer::ICUCleanUp();
  }
  catch(...)
  {
    cerr << "An unknown error occurred in creation of module's xml file!" << endl;
  }
  // generate the xml file
  ////////////////////////
  try
  {
    XMLPlatformUtils::Initialize();
    XalanTransformer::initialize();
    XalanTransformer theXalanTransformer;
    theResult = theXalanTransformer.transform(xml_source.c_str(), this->xml_xsl_.c_str(), xml_output.c_str());
    if(theResult != 0)
    {
      cerr << "Error: " << theXalanTransformer.getLastError() << endl;
    }
    XalanTransformer::terminate();
    XMLPlatformUtils::Terminate();
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

bool Translator::translate(DOMNode* xml_source, string cc_output, string gui_output, string xml_output)
{
  cerr << "Not implemented yet\n";
  return false;
}
