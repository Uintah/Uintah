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
  cc_xsl_file_ = "";
  gui_xsl_file_ = "";
  xml_xsl_file_ = "";
  cc_out_ = "";
  gui_out_ = "";
  xml_out_ = "";
  path_to_insight_package_ = "";

  parser_ = new Parser();
  translator_ = new Translator();
//  translator_ = new Translator(cc_xsl_file_, gui_xsl_file_, xml_xsl_file_);
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

void Generator::set_cc_xsl_file(string f)
{
  cc_xsl_file_ = f;
  this->translator_->set_cc_xsl(cc_xsl_file_);
}

string Generator::get_cc_xsl_file( void )
{
  return cc_xsl_file_;
}

void Generator::set_gui_xsl_file(string f)
{
  gui_xsl_file_ = f;
  this->translator_->set_gui_xsl(gui_xsl_file_);
}

string Generator::get_gui_xsl_file( void )
{
  return gui_xsl_file_;
}

void Generator::set_xml_xsl_file(string f)
{
  xml_xsl_file_ = f;
  this->translator_->set_xml_xsl(xml_xsl_file_);
}

string Generator::get_xml_xsl_file( void )
{
  return xml_xsl_file_;
}

void Generator::set_cc_out(string f)
{
  cc_out_ = f;
}

string Generator::get_cc_out( void )
{
  return cc_out_;
}

void Generator::set_gui_out(string f)
{
  gui_out_ = f;
}

string Generator::get_gui_out( void )
{
  return gui_out_;
}

void Generator::set_xml_out(string f)
{
  xml_out_ = f;
}

string Generator::get_xml_out( void )
{
  return xml_out_;
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

//////////////////////////////
// Start generator
//////////////////////////////
bool Generator::generate( void )
{
  if(this->xml_file_ == "") {
    cerr << "XML File not set\n";
    return false;
  }
  cout << this->get_xml_file() << endl;
  // parse file for includes and imports
  DOMNode* node = this->parser_->read_input_file(this->get_xml_file());

  if(node == 0)
    return false;

  // write out new xml file FIX
  string temp_file = "/tmp/mp.xml";
  std::ofstream out;
  out.open(temp_file.c_str());
  out << "<?xml version=\"1.0\"?>\n";

  this->parser_->write_node(out, node);

  out.close();

  // use xsl to translate
  this->translator_->translate(temp_file, this->cc_out_, this->gui_out_, this->xml_out_);

  return true;
}
