
#ifndef PackageParser_h
#define PackageParser_h

#include <Dataflow/Resources/ResourcesXMLParser.h>

namespace SCIRun {

class ModuleParser;
class PortParser;
class PackageInfo;

class PackageParser : public ResourcesXMLParser {
protected:
  PackageInfo *package_;
  ModuleParser *module_parser_;
  PortParser *port_parser_;

public:

  PackageParser( Resources * );
  virtual ~PackageParser();

  bool parse( PackageInfo * );
  

  virtual void endElement (const XMLCh* const uri,
			   const XMLCh* const localname,
			   const XMLCh* const qname);
};

} // namespace SCIRun

#endif

