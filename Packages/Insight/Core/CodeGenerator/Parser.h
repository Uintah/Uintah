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


private:

  string path_to_insight_package_;

};
