/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
