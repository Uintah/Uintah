
#ifndef XMLParser_h
#define XMLParser_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#pragma set woff 3201
#endif

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#pragma reset woff 3201
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
