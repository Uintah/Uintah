/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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
