#include <iostream>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Resources/PortParser.h>
#include <Dataflow/Resources/Resources.h>

namespace SCIRun {

PortParser::PortParser( Resources *s )
  : ResourcesXMLParser(s)
{
}

PortParser::~PortParser()
{
}

void 
PortParser::startElement( const XMLCh * const uri,
			    const XMLCh * const localname,
			    const XMLCh * const qname,
			    const Attributes&   attrs )
{
  static XMLCh *t_name = XMLString::transcode("name");

  string tag ( XMLString::transcode(localname) );
  if ( tag == "port" ) {
    string name = XMLString::transcode(attrs.getValue( t_name ) );
    info_ = scinew PortInfo;
    info_->type_ = package_->name_ + "::" + name;
    info_->package_ = package_->name_;
    info_->imaker_ = "make_" + name + "IPort";
    info_->omaker_ = "make_" + name + "OPort";
    cerr << "Read port " << info_->type_ << endl;
    resources_->ports_[info_->type_] = info_;
  }

  XMLParser::startElement( uri, localname, qname, attrs );
}
  
void
PortParser::endElement (const XMLCh* const uri,
			const XMLCh* const localname,
			const XMLCh* const qname)
{
  string tag ( XMLString::transcode(localname) );
  if ( tag == "datatype" )
    info_->datatype_ = data_;
  else if ( tag == "imaker" )
    info_->imaker_ = data_;
  else if (tag == "omaker_" )
    info_->omaker_ = data_;
  else if ( tag == "lib" )
    info_->libs_.push_back( data_);
  else if ( tag == "port" ) {
    info_->libs_.push_back( package_->lib_path_+"_Ports.so" );
    info_->libs_.push_back( package_->lib_path_+".so");
  }
};

} // namespace SCIRun 
