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
