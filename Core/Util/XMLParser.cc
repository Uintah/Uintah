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



#include <Core/Util/XMLParser.h>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 3303
#pragma set woff 3201
#pragma set woff 1209
#pragma set woff 1110
#endif
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 3303
#pragma reset woff 3201
#pragma reset woff 1209
#pragma reset woff 1110
#endif
#include <iostream>
using std::cout;
using std::endl;

namespace SCIRun {

bool XMLParser::initialized_ = false;

XMLParser::XMLParser( )
{
  if ( !initialized_ )
    try {
      XMLPlatformUtils::Initialize();
      initialized_ = true;
    }
    catch (const XMLException& e) {
      cout << "Error during initialization! :\n"
	   << e.getMessage() << "\n";
    }
  
  reader_ = XMLReaderFactory::createXMLReader();
  init();
}

XMLParser::~XMLParser()
{
  //  delete reader_;
}

void
XMLParser::init()
{
  reader_->setContentHandler( this );
  reader_->setErrorHandler( this );
}

bool
XMLParser::parse( const string &file )
{
  bool ok = false;

  try {
    reader_->parse( file.c_str() );
    ok = true;
  }
  catch ( const XMLException &exception ) {
    cout << "XMLParser error: " <<  
      XMLString::transcode(exception.getMessage()) << endl;
  }

  return ok;
}

void
XMLParser:: startElement(const XMLCh * const /*uri*/,
			 const XMLCh * const /*localname*/,
			 const XMLCh * const /*qname*/,
			 const Attributes&   /*attrs*/ ) 
{
  data_ = "";
}

void
XMLParser::characters( const XMLCh *const chars,
			  const unsigned int length )
{
  data_.append( XMLString::transcode(chars), length );
}

} // namespace SCIRun





