

#include <Core/Util/XMLParser.h>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <sax2/SAX2XMLReader.hpp>
#include <sax2/XMLReaderFactory.hpp>

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
XMLParser:: startElement(const XMLCh * const uri,
			 const XMLCh * const localname,
			 const XMLCh * const qname,
			 const Attributes&   attrs ) 
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





