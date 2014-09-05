#include <iostream>

#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Resources/Resources.h>
#include <Dataflow/Resources/ResourcesParser.h>
#include <Dataflow/Resources/PackageParser.h>

namespace SCIRun {

ResourcesParser::ResourcesParser( Resources *s )
  : ResourcesXMLParser(s)
{
  package_parser_ = scinew PackageParser(s);
  mode_.push(NoMode);
}

ResourcesParser::~ResourcesParser()
{
}

void 
ResourcesParser::startElement( const XMLCh * const uri,
			       const XMLCh * const localname,
			       const XMLCh * const qname,
			       const Attributes&   attrs )
{
  string tag ( XMLString::transcode(localname) );

  if ( tag == "packages" )
    mode_.push(PackagesMode);
  else if ( tag == "package" ) {
    mode_.push(PackageMode);
    package_ = scinew PackageInfo;
    package_->level_ = 1;
  } else if ( tag == "data" )
    mode_.push(DataMode);

  ResourcesXMLParser::startElement( uri, localname, qname, attrs );
}

void
ResourcesParser::endElement (const XMLCh* const uri,
			     const XMLCh* const localname,
			     const XMLCh* const qname)
{
  string tag ( XMLString::transcode(localname) );

  switch ( mode_.top() ) {
  case PackageMode:
    if ( tag == "name" ) {
      package_->name_ = data_;
      if ( data_ == "SCIRun" ) {
	package_->path_ = "../src/Dataflow/XML";
	package_->lib_path_ = "lib/libDataflow";
	package_->ui_path_ = "../src/Dataflow/GUI";
      }
      else {
	package_->path_ = string("../src/") + data_ + "Dataflow/XML";
	package_->lib_path_ = string("lib/lib") + data_; 
	package_->ui_path_ = string("../src/") + data_ + "Dataflow/GUI";
      }
    }
    else if ( tag == "path" ) 
      package_->path_ = data_[0] == '/' ? data_ : (string("../src/")+data_);
    else if ( tag == "level" ) 
      string_to_int( data_, package_->level_ );
    else if ( tag == "package" ) {
      resources.packages_[package_->name_] = package_;
      package_parser_->parse( package_ );
      mode_.pop();
    }
    break;
  case PackagesMode:
    if ( tag == "packages" )
      mode_.pop();
    break;
  case DataMode:
    if ( tag == "path" )
      resources.data_path_ = data_;
    break;
  default:
    cerr << "known mode in ResourcesParser\n";
    break;
  }
}

} // namespace SCIRun
