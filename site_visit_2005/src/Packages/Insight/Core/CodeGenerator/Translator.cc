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
