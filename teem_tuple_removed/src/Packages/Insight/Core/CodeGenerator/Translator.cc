/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
