/* REFERENCED */
static char *id="@(#) $Id$";

#include "ProblemSpec.h"

#include <iostream>
#include <cstdlib>


using std::cerr;

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

  DOM_Node start_element = prob_spec->d_doc.getDocumentElement();
  DOM_Node found_node = findNode(name,start_element);

  if (found_node.isNull()) {
    cerr << "Didn't find the tag . . " << endl;
    cerr << "Setting to Null . . " << endl;
    prob_spec = 0;
  }
  else {
    prob_spec->setNode(found_node);
    DOMString found_node_name = found_node.getNodeName();
    found_node_name.println();
  }
  
  return prob_spec;

}


DOM_Node ProblemSpec::findNode(const std::string &name,DOM_Node node) const
{

  // Convert string name to a DOMString;
  
  DOMString search_name(name.c_str());
  DOM_Node child = node.getFirstChild();
  while (child != 0) {
    DOMString child_name = child.getNodeName();
    if (child_name.equals(search_name) ){
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
	int cmp = atoi(s);
	delete[] s;
	if (cmp == 0)
	   value = false;
	else if (cmp == 1)
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
	value = string(s);
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
