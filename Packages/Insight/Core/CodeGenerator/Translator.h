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

#include "Enumeration.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

class Translator {

public:
  Translator();
  Translator(string xsl);
  ~Translator();


  void set_xsl_file(string);
  string get_xsl_file();

  bool translate(string xml_source, string output, FileFormat format);

  bool translate(DOMNode* xml_source, string output, FileFormat format);

private:
  string xsl_;
};
