
/* REFERENCED */
static char *id="@(#) $Id$";

#include "ProblemSpec.h"

#include <iostream>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <SCICore/Malloc/Allocator.h>
//#include <cstdlib>
using namespace Uintah;
using namespace std;
using namespace SCICore::Geometry;
using namespace PSECore::XMLUtil;

ProblemSpec::ProblemSpec(const DOM_Node& node)
   : d_node(node)
{
}

ProblemSpec::~ProblemSpec()
{
}

ProblemSpecP ProblemSpec::findBlock() const
{
  ProblemSpecP prob_spec = scinew ProblemSpec(d_node);

  DOM_Node child = d_node.getFirstChild();
  if (child != 0) {
    if (child.getNodeType() == DOM_Node::TEXT_NODE) {
      child = child.getNextSibling();
    }
  }
  if (child.isNull() )
     return 0;
  else
     return scinew ProblemSpec(child);

}

ProblemSpecP ProblemSpec::findBlock(const std::string& name) const 
{
   DOM_Node found_node = findNode(name,d_node);

  if (found_node.isNull()) {
    cerr << "Didn't find the tag . . " << name << endl;
    cerr << "Setting to Null . . " << endl;
    return 0;
  }
  else {
     return scinew ProblemSpec(found_node);
  }
}

ProblemSpecP ProblemSpec::findNextBlock() const
{
  DOM_Node found_node = d_node.getNextSibling();
  
  if (found_node != 0) {
    if (found_node.getNodeType() == DOM_Node::TEXT_NODE) {
      found_node = found_node.getNextSibling();
    }
  }
    
  if (found_node.isNull() ) {
     return 0;
  }
  else {
     return scinew ProblemSpec(found_node);
  }
}

ProblemSpecP ProblemSpec::findNextBlock(const std::string& name) const 
{
  // Iterate through all of the child nodes that have this name

  DOM_Node found_node = d_node.getNextSibling();

  DOMString search_name(name.c_str());
  while(found_node != 0){
    DOMString node_name = found_node.getNodeName();
    char *s = node_name.transcode();
    std::string c_name(s);
    delete[] s;
    if (search_name.equals(node_name) ) {
      break;
    }
    //DOM_Node tmp = findNode(name,child);
    found_node = found_node.getNextSibling();

  }
   
  if (found_node.isNull()) {
     return 0;
  }
  else {
     return scinew ProblemSpec(found_node);
  }
}


std::string ProblemSpec::getNodeName() const
{

  DOMString node_name = d_node.getNodeName();
  char *s = node_name.transcode();
  std::string name(s);
  delete[] s;

  return name;
}

ProblemSpecP ProblemSpec::get(const std::string& name, double &value)
{
  ProblemSpecP ps = this;

  DOM_Node found_node = findNode(name,this->d_node);
  if (found_node.isNull()) {
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

ProblemSpecP ProblemSpec::get(const std::string& name, 
			      Point &value)
{
    Vector v;
    ProblemSpecP ps = get(name, v);
    value = Point(v);
    return ps;
}

ProblemSpecP ProblemSpec::get(const std::string& name, 
			      Vector &value)
{

  std::string string_value;
  ProblemSpecP ps = this;
  DOM_Node found_node = findNode(name, this->d_node);
  if (found_node.isNull()) {
    ps = 0;
    return ps;
  }
  else {
    for (DOM_Node child = found_node.getFirstChild(); child != 0;
	 child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	DOMString val = child.getNodeValue();
	char *s = val.transcode();
	string_value = std::string(s);
	delete[] s;
	// Parse out the [num,num,num]
	// Now pull apart the string_value
	std::string::size_type i1 = string_value.find("[");
	std::string::size_type i2 = string_value.find_first_of(",");
	std::string::size_type i3 = string_value.find_last_of(",");
	std::string::size_type i4 = string_value.find("]");
	
	std::string x_val(string_value,i1+1,i2-i1-1);
	std::string y_val(string_value,i2+1,i3-i2-1);
	std::string z_val(string_value,i3+1,i4-i3-1);

	value.x(atof(x_val.c_str()));
	value.y(atof(y_val.c_str()));
	value.z(atof(z_val.c_str()));	
      }
    }
  }
          
  return ps;

}

ProblemSpecP ProblemSpec::get(const std::string& name, 
			      IntVector &value)
{

  std::string string_value;
  ProblemSpecP ps = this;
  DOM_Node found_node = findNode(name, this->d_node);
  if (found_node.isNull()) {
    ps = 0;
    return ps;
  }
  else {
    for (DOM_Node child = found_node.getFirstChild(); child != 0;
	 child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	DOMString val = child.getNodeValue();
	char *s = val.transcode();
	string_value = std::string(s);
	delete[] s;
	// Parse out the [num,num,num]
	// Now pull apart the string_value
	std::string::size_type i1 = string_value.find("[");
	std::string::size_type i2 = string_value.find_first_of(",");
	std::string::size_type i3 = string_value.find_last_of(",");
	std::string::size_type i4 = string_value.find("]");
	
	std::string x_val(string_value,i1+1,i2-i1-1);
	std::string y_val(string_value,i2+1,i3-i2-1);
	std::string z_val(string_value,i3+1,i4-i3-1);
			
	value.x(atoi(x_val.c_str()));
	value.y(atoi(y_val.c_str()));
	value.z(atoi(z_val.c_str()));	
      }
    }
  }
          
  return ps;

}

void ProblemSpec::require(const std::string& name, double& value)
{

  // Check if the prob_spec is NULL

  if (! this->get(name,value))
      throw ParameterNotFound(name);
 
}

void ProblemSpec::require(const std::string& name, int& value)
{

 // Check if the prob_spec is NULL

  if (! this->get(name,value))
      throw ParameterNotFound(name);
  
}

void ProblemSpec::require(const std::string& name, bool& value)
{
 // Check if the prob_spec is NULL

  if (! this->get(name,value))
      throw ParameterNotFound(name);
 

}

void ProblemSpec::require(const std::string& name, std::string& value)
{
 // Check if the prob_spec is NULL

  if (! this->get(name,value))
      throw ParameterNotFound(name);
 
}

void ProblemSpec::require(const std::string& name, 
			  Vector  &value)
{

  // Check if the prob_spec is NULL

 if (! this->get(name,value))
      throw ParameterNotFound(name);

}

void ProblemSpec::require(const std::string& name, 
			  IntVector  &value)
{

  // Check if the prob_spec is NULL

 if (! this->get(name,value))
      throw ParameterNotFound(name);

}

void ProblemSpec::require(const std::string& name, 
			  Point  &value)
{

  // Check if the prob_spec is NULL

 if (! this->get(name,value))
      throw ParameterNotFound(name);

}


ProblemSpecP ProblemSpec::getOptional(const std::string& name, 
				      std::string &value)
{
  ProblemSpecP ps = this;
  DOM_Node attr_node;
  DOM_Node found_node = findNode(name,this->d_node);
  if (found_node.isNull()) {
    ps = 0;
    return ps;
  }
  else {
    if (found_node.getNodeType() == DOM_Node::ELEMENT_NODE) {
      // We have an element node and attributes
      DOM_NamedNodeMap attr = found_node.getAttributes();
      int num_attr = attr.getLength();
      if (num_attr >= 1)
	attr_node = attr.item(0);
      else 
	return ps = 0;
      if (attr_node.getNodeType() == DOM_Node::ATTRIBUTE_NODE) {
	DOMString val = attr_node.getNodeValue();
	char *s = val.transcode();
	value = std::string(s);
	delete[] s;
      }
      else {
	ps = 0;
      }
    }
  }
          
  return ps;

}

void ProblemSpec::requireOptional(const std::string& name, std::string& value)
{
 // Check if the prob_spec is NULL

  if (! this->getOptional(name,value))
      throw ParameterNotFound(name);
 
}


const TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}

//
// $Log$
// Revision 1.19  2000/05/30 20:19:41  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.18  2000/05/20 08:09:38  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.17  2000/04/27 21:26:37  jas
// Fixed the parsing of Vector and IntVector, offsets were wrong for the
// y and z values.
//
// Revision 1.16  2000/04/27 00:28:37  jas
// Can now read the attributes field of a tag.
//
// Revision 1.15  2000/04/26 06:49:11  sparker
// Streamlined namespaces
//
// Revision 1.14  2000/04/20 23:57:03  jas
// Fixed the findBlock() and findNextBlock() to iterate through all the
// nodes.  Now we can go thru the MPM setup without an error.  There is
// still the problem of where the <res> tag should go.
//
// Revision 1.13  2000/04/20 22:37:17  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
//
// Revision 1.12  2000/04/19 05:26:18  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.11  2000/04/13 06:51:05  sparker
// More implementation to get this to work
//
// Revision 1.10  2000/04/12 23:01:55  sparker
// Implemented more of problem spec - added Point and IntVector readers
//
// Revision 1.9  2000/04/12 15:33:49  jas
// Can now read a Vector type [num,num,num] from the ProblemSpec.
//
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
