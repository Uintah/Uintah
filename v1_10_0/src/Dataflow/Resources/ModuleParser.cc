#include <iostream>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Resources/ModuleParser.h>
#include <Dataflow/Resources/Resources.h>

namespace SCIRun {


ModuleParser::ModuleParser( Resources *s )
  : ResourcesXMLParser(s)
{
  mode_.push( NoMode );
}

ModuleParser::~ModuleParser()
{
}

void 
ModuleParser::startElement( const XMLCh * const uri,
			    const XMLCh * const localname,
			    const XMLCh * const qname,
			    const Attributes&   attrs )
{
  static XMLCh *t_name = XMLString::transcode("name");
  static XMLCh *t_cat = XMLString::transcode("category");
  static XMLCh *t_dynamic = XMLString::transcode("lastportdynamic");

  string tag ( XMLString::transcode(localname) );

  if ( tag == "component" ) {
    info_ = scinew ModuleInfo;
    info_->package_ = package_->name_;
    info_->name_ = XMLString::transcode(attrs.getValue( t_name ) );
    info_->maker_ = "make_"+ info_->name_;
    info_->ui_ = info_->name_+".tcl";

    // backward compatibility
    string cat = XMLString::transcode(attrs.getValue(t_cat));
    info_->categories_.push_back(cat);
    info_->libs_.push_back( package_->lib_path_+"_Modules_"+ cat + ".so" );

    info_->id_ = package_->name_+ "_" + cat+"_" + info_->name_;
    cerr <<"module " << info_->id_ << endl;

    resources_->modules_[info_->id_] = info_;
  }
  else if ( tag == "inputs" ) {
    port_mode_ = InputMode;
    string dynamic = "no";
    const XMLCh *d = attrs.getValue( t_dynamic );
    if ( d ) dynamic = XMLString::transcode(d);
    info_->has_dynamic_port_ = dynamic == "yes";
  }
  else if ( tag == "outputs" )
    port_mode_ = OutputMode;
  else if ( tag == "io" ) 
    mode_.push(IoMode);
  else if ( tag == "port" ) 
    port_info_ = scinew ModulePortInfo;

  XMLParser::startElement( uri, localname, qname, attrs );
}
  
void
ModuleParser::endElement (const XMLCh* const uri,
			  const XMLCh* const localname,
			  const XMLCh* const qname)
{
  string tag ( XMLString::transcode(localname) );

  switch (mode_.top()) {
  case IoMode:
    if ( tag == "name" ) 
      port_info_->name_ = data_;
    else if ( tag == "datatype" ) 
      port_info_->type_ = data_;
    else if ( tag == "port" ) {
      cerr << "add port: " << port_info_->name_ << " " << port_info_->type_ << endl;
      if ( port_mode_  == InputMode )
	info_->iports_.push_back( port_info_ );
      else
	info_->oports_.push_back( port_info_ );
      port_info_ = 0;
    } else if ( tag == "io")
      mode_.pop();
    break;
  default:
    if ( tag == "maker" )
      info_->maker_ = data_;
    else if ( tag == "lib" ) 
      info_->libs_.push_back( package_->lib_path_+"_"+data_+".so"  );
    else if ( tag == "module" ) {
      info_->libs_.push_back( package_->lib_path_+".so"); 
      cerr << "libs: \n" << "\t"<< info_->libs_[0] << endl;
      cerr << "        " << "\t"<< info_->libs_[1] << endl;

    }
    break;
  }      
}

} // namespace SCIRun
