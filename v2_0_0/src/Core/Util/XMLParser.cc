

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





