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
