
#ifndef XMLParser_h
#define XMLParser_h

#include <string>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif

#include <util/PlatformUtils.hpp>
#include <sax/ErrorHandler.hpp>
#include <sax2/Attributes.hpp>
#include <sax2/DefaultHandler.hpp>

#ifdef __sgi
#pragma reset woff 1375
#endif


class SAX2XMLReader;

namespace SCIRun {

using namespace std;

class XMLParser : public DefaultHandler {
private:
  static bool initialized_;

protected:
  string data_;
  SAX2XMLReader* reader_;

public:

  XMLParser();
  virtual ~XMLParser();
  virtual void init();

  virtual bool parse( const string &file );

  virtual void startElement( const XMLCh * const uri,
			     const XMLCh * const localname,
			     const XMLCh * const qname,
			     const Attributes&   attrs );

  virtual void characters(  const XMLCh *const, const unsigned int );
};

} // namespace SCIRun

#endif
