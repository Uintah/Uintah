
#ifndef ModuleParser_h
#define ModuleParser_h

#include <stack>
#include <Dataflow/Resources/ResourcesXMLParser.h>

namespace SCIRun {

class ModuleInfo;
class ModulePortInfo;
class PackageInfo;

class ModuleParser : public ResourcesXMLParser {
private:
  typedef enum {NoMode, IoMode, InputMode, OutputMode } Mode;

protected:
  stack<Mode> mode_;
  Mode port_mode_;
  ModuleInfo *info_;
  ModulePortInfo *port_info_;

public:

  ModuleParser( Resources * );
  virtual ~ModuleParser();

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

