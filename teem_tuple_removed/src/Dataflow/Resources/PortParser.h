
#ifndef PortParser_h
#define PortParser_h

#include <stack>
#include <Dataflow/Resources/ResourcesXMLParser.h>

namespace SCIRun {

class PortInfo;

class PortParser : public ResourcesXMLParser {
protected:
  PortInfo *info_;

public:
  PortParser( Resources * );
  virtual ~PortParser();

  virtual void startElement( const XMLCh * const uri,
			     const XMLCh * const localname,
			     const XMLCh * const qname,
			     const Attributes&   attrs );
  
  virtual void endElement (const XMLCh* const uri,
			   const XMLCh* const localname,
			   const XMLCh* const qname);
};

} // namespace SCIRun

#endif

