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
 * Parser.h
 *
 * Written by:
 *   Darby J Van Uitert
 *   SCI Institute
 *   January 2003
 *************************************/

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMException.hpp> 

#include <Dataflow/XMLUtil/SimpleErrorHandler.h>

#include <iostream>
#include <fstream.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ostream;

class Parser {

public:
  Parser();
  ~Parser();

  // parse the file for includes and imports
  DOMNode* read_input_file(string filename);
  
  // resolve <include> and <import> tags
  void resolve_includes(DOMNode* );
  void resolve_imports(DOMNode* );

  // output tree
  void write_node(ostream&, DOMNode*);
  void output_content(ostream&,  string chars );

  void set_path_to_insight_package(string f);
  string get_path_to_insight_package( void );


  bool has_errors_;
private:
  
  string path_to_insight_package_;

};
