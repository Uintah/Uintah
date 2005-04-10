#ifndef PSECore_Dataflow_PackageDBHandler_h
#define PSECore_Dataflow_PackageDBHandler_h 1

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <sax/ErrorHandler.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

namespace PSECore {
namespace Dataflow {

class PackageDBHandler : public ErrorHandler
{
public:
  bool foundError;
  
  PackageDBHandler();
  ~PackageDBHandler();
  
  void warning(const SAXParseException& e);
  void error(const SAXParseException& e);
  void fatalError(const SAXParseException& e);
  void resetErrors();
  
private :
  PackageDBHandler(const PackageDBHandler&);
  void operator=(const PackageDBHandler&);
};

} // Dataflow
} // PSECore

#endif

