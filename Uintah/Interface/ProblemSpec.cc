/* REFERENCED */
static char *id="@(#) $Id$";

#include "ProblemSpec.h"

#include <iostream>
//#include <cstdlib>

using std::cerr;
using std::endl;

namespace Uintah {
namespace Interface {

ProblemSpec::ProblemSpec()
{
}

ProblemSpec::~ProblemSpec()
{
}

void ProblemSpec::setDoc(const DOM_Document &doc)
{
  d_doc = doc;
}

void ProblemSpec::setNode(const DOM_Node &node)
{
  d_node = node;
}

ProblemSpecP ProblemSpec::findBlock(const std::string& name) const 
{
  ProblemSpecP prob_spec = new ProblemSpec;
  prob_spec->setNode(this->d_node);
  prob_spec->setDoc(this->d_doc);

  DOM_Node start_element;
  if (d_node.isNull()) {
    start_element = prob_spec->d_doc.getDocumentElement();
  }
  else {
    start_element = this->d_node;
  }
  DOM_Node found_node = findNode(name,start_element);

  if (found_node.isNull()) {
    cerr << "Didn't find the tag . . " << name << endl;
    cerr << "Setting to Null . . " << endl;
    prob_spec = 0;
  }
  else {
    prob_spec->setNode(found_node);
  }
  
  return prob_spec;

}

ProblemSpecP ProblemSpec::findNextBlock(const std::string& name) const 
{
  ProblemSpecP prob_spec = new ProblemSpec;
  prob_spec->setNode(this->d_node);
  prob_spec->setDoc(this->d_doc);

  DOM_Node start_element;
  if (d_node.isNull()) {
    start_element = prob_spec->d_doc.getDocumentElement();
  }
  else {
    start_element = this->d_node;
  }

  // Iterate through all of the child nodes that have this name

  DOM_Node found_node = start_element.getNextSibling();
 
  if (!found_node.isNull()) {
    if (found_node.getNodeType() == DOM_Node::TEXT_NODE) {
      found_node = found_node.getNextSibling();
    }
  }
   
  if (found_node.isNull()) {
    prob_spec = 0;
  }
  else {
    prob_spec->setNode(found_node);
  }
  
  return prob_spec;

}


DOM_Node ProblemSpec::findNode(const std::string &name,DOM_Node node) const
{

  // Convert string name to a DOMString;
  
  DOMString search_name(name.c_str());
  // Check if the node is equal
  DOMString node_name = node.getNodeName();
  if (node_name.equals(search_name))
    return node;
      
  // Do the child nodes now
  DOM_Node child = node.getFirstChild();
  while (child != 0) {
    DOMString child_name = child.getNodeName();
    char *s = child_name.transcode();
    std::string c_name(s);
    delete[] s;
    if (search_name.equals(child_name) ) {
      return child;
    }
    DOM_Node tmp = findNode(name,child);
    child = child.getNextSibling();
  }
  
  DOM_Node unknown;
  return unknown;
}


ProblemSpecP ProblemSpec::get(const std::string& name, double &value)
{
  ProblemSpecP ps = this;

  DOM_Node found_node = findNode(name,this->d_node);
  if (found_node.isNull()) {
    cerr << "Didn't find the tag . ." << endl;
    cerr << "Setting to Null . . " << endl;
    ps = 0;
    return ps;
  }
  else {
    for (DOM_Node child = found_node.getFirstChild(); child != 0;
	 child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	DOMString val = child.getNodeValue();
	char* s = val.transcode();
	value = atof(s);
	delete[] s;
      }
    }
  }
          
  return ps;
}

ProblemSpecP ProblemSpec::get(const std::string& name, int &value)
{
  ProblemSpecP ps = this;
  DOM_Node found_node = findNode(name,this->d_node);
  if (found_node.isNull()) {
    cerr << "Didn't find the tag . ." << endl;
    cerr << "Setting to Null . . " << endl;
    ps = 0;
    return ps;
  }
  else {
    for (DOM_Node child = found_node.getFirstChild(); child != 0;
	 child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	DOMString val = child.getNodeValue();
	char* s = val.transcode();
	value = atoi(s);
	delete[] s;
      }
    }
  }
          
  return ps;

}

ProblemSpecP ProblemSpec::get(const std::string& name, bool &value)
{
  ProblemSpecP ps = this;
  DOM_Node found_node = findNode(name,this->d_node);
  if (found_node.isNull()) {
    cerr << "Didn't find the tag . ." << endl;
    cerr << "Setting to Null . . " << endl;
    ps = 0;
    return ps;
  }
  else {
    for (DOM_Node child = found_node.getFirstChild(); child != 0;
	 child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	DOMString val = child.getNodeValue();
	char *s = val.transcode();
	std::string cmp(s);
	delete[] s;
	if (cmp == "false")
	   value = false;
	else if (cmp == "true")
	  value = true;
	
      }
    }
  }
          
  return ps;

}

ProblemSpecP ProblemSpec::get(const std::string& name, std::string &value)
{
  ProblemSpecP ps = this;
  DOM_Node found_node = findNode(name,this->d_node);
  if (found_node.isNull()) {
    cerr << "Didn't find the tag . ." << endl;
    cerr << "Setting to Null . . " << endl;
    ps = 0;
    return ps;
  }
  else {
    for (DOM_Node child = found_node.getFirstChild(); child != 0;
	 child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	DOMString val = child.getNodeValue();
	char *s = val.transcode();
	value = std::string(s);
	delete[] s;
      }
    }
  }
          
  return ps;

}

void ProblemSpec::require(const std::string& name, double& value)
{

  // Check if the prob_spec is NULL

  if (! this->get(name,value))
    cerr << "Throw an exception . . " << endl;
 
}

void ProblemSpec::require(const std::string& name, int& value)
{

 // Check if the prob_spec is NULL

  if (! this->get(name,value))
    cerr << "Throw an exception . . " << endl;
  
}

void ProblemSpec::require(const std::string& name, bool& value)
{
 // Check if the prob_spec is NULL

  if (! this->get(name,value))
    cerr << "Throw an exception . . " << endl;
 

}

void ProblemSpec::require(const std::string& name, std::string& value)
{
 // Check if the prob_spec is NULL

  if (! this->get(name,value))
    cerr << "Throw an exception . . " << endl;
 
}


const TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}

} // end namespace Interface 
} // end namespace Uintah

//
// $Log$
// Revision 1.8  2000/04/07 18:40:51  jas
// Fixed bug in getNextBlock.
//
// Revision 1.7  2000/04/06 02:33:33  jas
// Added findNextBlock which will find all of the tags named name within a
// given block.
//
// Revision 1.6  2000/03/31 17:52:08  moulding
// removed #include <cstdlib> and changed line 178 from "value = string(s);" to
// "value = std::string(s);"
//
// Revision 1.5  2000/03/31 00:55:07  jas
// Let the include stuff know we are using endl.
//
// Revision 1.4  2000/03/29 23:48:00  jas
// Filled in methods for extracting data from the xml tree (requires and get)
// and storing the result in a variable.
//
// Revision 1.3  2000/03/29 01:59:59  jas
// Filled in the findBlock method.
//
// Revision 1.2  2000/03/23 20:00:17  jas
// Changed the include files, namespace, and using statements to reflect the
// move of ProblemSpec from Grid/ to Interface/.
//
// Revision 1.1  2000/03/23 19:47:55  jas
// Moved the ProblemSpec stuff from Grid/ to Interface.
//
// Revision 1.3  2000/03/21 18:52:11  sparker
// Prototyped header file for new problem spec functionality
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
