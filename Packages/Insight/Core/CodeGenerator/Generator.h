/**************************************
 *
 * Generator.h
 *
 * Written by:
 *   Darby J Van Uitert
 *   SCI Institute
 *   January 2003
 *************************************/

#include "Parser.h"
#include "Translator.h"
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

class Generator {

public:
  Generator();
  ~Generator();

  // get/set files
  void set_xml_file(string f);
  string get_xml_file( void );
  void set_cc_xsl_file(string f);
  string get_cc_xsl_file( void );
  void set_gui_xsl_file(string f);   
  string get_gui_xsl_file( void );
  void set_xml_xsl_file(string f);
  string get_xml_xsl_file( void );
  void set_cc_out(string f);
  string get_cc_out( void );
  void set_gui_out(string f);
  string get_gui_out( void );
  void set_xml_out(string f);
  string get_xml_out( void );
  void set_path_to_insight_package(string f);
  string get_path_to_insight_package( void );

  // execute the generator
  bool generate( void );

private:
  Parser* parser_;
  Translator* translator_;

  string xml_file_;
  string cc_xsl_file_;
  string gui_xsl_file_;
  string xml_xsl_file_;
  string cc_out_;
  string gui_out_;
  string xml_out_;
  string path_to_insight_package_;
};
