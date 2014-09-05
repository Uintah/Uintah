
#ifndef ResroucesParser_h
#define ResroucesParser_h

#include <stack>

#include <Dataflow/Resources/ResourcesXMLParser.h>

namespace SCIRun {

using namespace std;

class PackageParser;
class PackageInfo;


class ResourcesParser : public ResourcesXMLParser {
private:
  typedef enum {NoMode, PackagesMode, PackageMode, DataMode } Modes;

  PackageParser* package_parser_;
  
  stack<Modes> mode_;
  PackageInfo *package_;

public:
  ResourcesParser( Resources * );
  virtual ~ResourcesParser();

  void startElement( const XMLCh * const uri,
		     const XMLCh * const localname,
		     const XMLCh * const qname,
		     const Attributes&   attrs );

  void endElement (const XMLCh* const uri,
		   const XMLCh* const localname,
		   const XMLCh* const qname);
};

} // namespace SCIRun

#endif

