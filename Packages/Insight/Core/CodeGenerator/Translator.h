/**************************************
 *
 * Translator.h
 *
 * Written by:
 *   Darby J Van Uitert
 *   SCI Institute
 *   January 2003
 *************************************/

#include <Include/PlatformDefinitions.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <XalanTransformer/XalanTransformer.hpp>

#include <xercesc/dom/DOMNode.hpp>

#include <iostream>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

class Translator {

public:
  Translator();
  Translator(string, string, string);
  ~Translator();


  void set_cc_xsl(string);
  string get_cc_xsl();
  void set_gui_xsl(string);
  string get_gui_xsl();
  void set_xml_xsl(string);
  string get_xml_xsl();


  bool translate(string xml_source, string cc_output, string gui_output, string xml_output);

  bool translate(DOMNode* xml_source, string cc_output, string gui_output, string xml_output);

private:
  string cc_xsl_;
  string gui_xsl_;
  string xml_xsl_;
};
