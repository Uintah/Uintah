

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Resources/Resources.h>
#include <Dataflow/Resources/PackageParser.h>
#include <Dataflow/Resources/ModuleParser.h>
#include <Dataflow/Resources/PortParser.h>

namespace SCIRun {

PackageParser::PackageParser( Resources *s )
  : ResourcesXMLParser(s)
{
  module_parser_ = scinew ModuleParser(s);
  port_parser_ = scinew PortParser(s);
}

PackageParser::~PackageParser()
{
}

bool
PackageParser::parse( PackageInfo *info )
{
  package_ = info;
  module_parser_->set_package( package_ ); 
  port_parser_->set_package( package_ );

  return XMLParser::parse( package_->path_ + "/package.xml" );
}

void
PackageParser::endElement (const XMLCh* const uri,
			   const XMLCh* const localname,
			   const XMLCh* const qname)
{
  string tag ( XMLString::transcode(localname) );

  if ( tag == "module" )
    module_parser_->parse( package_->path_ + "/" + data_ + ".xml" );
  else if ( tag == "port" ) 
    port_parser_->parse( package_->path_ + "/" + data_ + ".xml" );
  else if ( tag == "ui_path" )
    package_->ui_path_ = data_;
}

} // namespace SCIRun
