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


/**************************************
 *
 * Generator.cc
 *
 * Written by:
 *   Darby J Van Uitert
 *   SCI Institute
 *   January 2003
 *************************************/

#include "Generator.h"
#include <fstream>

///////////////////////////////
// Constructors and Destructors
///////////////////////////////
Generator::Generator()
{
  xml_file_ = "";
  xsl_file_ = "";
  output_file_ = "";
  path_to_insight_package_ = "";

  parser_ = new Parser();
  translator_ = new Translator();
}


Generator::~Generator()
{

}

/////////////////////////////
// Get/Set methods for files
/////////////////////////////
void Generator::set_xml_file(string f)
{
  xml_file_ = f;
}

string Generator::get_xml_file( void )
{
  return xml_file_;
}

void Generator::set_xsl_file(string f)
{
  xsl_file_ = f;
  this->translator_->set_xsl_file(xsl_file_);
}

string Generator::get_xsl_file( void )
{
  return xsl_file_;
}

void Generator::set_output_file(string f)
{
  output_file_ = f;
}

string Generator::get_output_file( void )
{
  return output_file_;
}

void Generator::set_path_to_insight_package(string f)
{
  path_to_insight_package_ = f;
  this->parser_->set_path_to_insight_package(path_to_insight_package_);
}

string Generator::get_path_to_insight_package( void )
{
  return path_to_insight_package_;
}

void Generator::set_category(string f)
{
  category_ = f;
}

string Generator::get_category( void )
{
  return category_;
}

void Generator::set_module(string f)
{
  module_ = f;
}

string Generator::get_module( void )
{
  return module_;
}


//////////////////////////////
// Start generator
//////////////////////////////
bool Generator::generate( FileFormat format )
{
  if(this->xml_file_ == "") {
    cerr << "XML File not set\n";
    return false;
  }

  // parse file for include and import tags
  DOMNode* node = this->parser_->read_input_file(this->get_xml_file());

  if(node == 0)
    return false;

  if(parser_->has_errors_) 
    return false;
  
  // Generate temporary file and write it out.
  // This file combines the sci, itk, and gui xml files
  // and is written out in the same XML file as the
  // sci.xml files
  string temp_file = get_path_to_insight_package();
  temp_file += "Dataflow/Modules/";
  temp_file += get_category();
  temp_file += "/XML/temp.xml";
  
  std::ofstream out;
  out.open(temp_file.c_str());
  out << "<?xml version=\"1.0\"?>\n";
  this->parser_->write_node(out, node);
  out.close();
  
  // generate the appropriate file
  this->translator_->translate(temp_file, this->output_file_, format);

  return true;
}
