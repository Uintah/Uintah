
#ifndef ResourcesXMLParser_h
#define ResourcesXMLParser_h

#include <Core/Util/XMLParser.h>

namespace SCIRun {

class Resources;
class PackageInfo;

class ResourcesXMLParser : public XMLParser {

protected:
  Resources *resources_;
  PackageInfo *package_;

public:

  ResourcesXMLParser( Resources * );
  virtual ~ResourcesXMLParser();

  void set_package( PackageInfo *p ) { package_ = p; }
};

} // namespace SCIRun

#endif

