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
