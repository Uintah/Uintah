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
#include "Enumeration.h"
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

  void set_xsl_file(string f);
  string get_xsl_file( void );

  void set_output_file(string f);
  string get_output_file( void );

  void set_path_to_insight_package(string f);
  string get_path_to_insight_package( void );

  void set_category(string f);
  string get_category( void );

  void set_module(string f);
  string get_module( void );

  // execute the generator
  bool generate( FileFormat format );

private:
  Parser* parser_;
  Translator* translator_;

  string xml_file_;
  string xsl_file_;
  string output_file_;
  string path_to_insight_package_;
  string category_;
  string module_;
};
