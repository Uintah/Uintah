
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <map>
#include <sstream>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/util/XMLString.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

using namespace Uintah;
using namespace SCIRun;

using namespace std;

ProblemSpec::ProblemSpec(const DOMNode* node, bool doWrite)
  // : d_node(const_cast<DOMNode*>(node)), d_write(doWrite)
  // pass node and not node->cloneNode instead of d_node(scinew DOMNode(node))
{
  d_node = const_cast<DOMNode*>(node);
  d_write = doWrite;
}
ProblemSpec::~ProblemSpec()
{
  // the problem spec doesn't allocate any new memory.
  //delete d_node;
}

ProblemSpecP ProblemSpec::findBlock() const
{
  //ProblemSpecP prob_spec = scinew ProblemSpec(d_node, d_write);

  const DOMNode* child = d_node->getFirstChild();
  if (child != 0) {
    if (child->getNodeType() == DOMNode::TEXT_NODE) {
      child = child->getNextSibling();
    }
  }
  if (child == NULL)
     return 0;
  else
     return scinew ProblemSpec(child, d_write);

}

ProblemSpecP ProblemSpec::findBlock(const std::string& name) const 
{
  const DOMNode* found_node = findNode(name,d_node);

  if (found_node == NULL) {
    return 0;
  }
  else {
    return scinew ProblemSpec(found_node, d_write);
  }
}

ProblemSpecP ProblemSpec::findNextBlock() const
{
  const DOMNode* found_node = d_node->getNextSibling();
  
  if (found_node != 0) {
    if (found_node->getNodeType() == DOMNode::TEXT_NODE) {
      found_node = found_node->getNextSibling();
    }
  }
    
  if (found_node == NULL ) {
     return 0;
  }
  else {
     return scinew ProblemSpec(found_node, d_write);
  }
}

ProblemSpecP ProblemSpec::findNextBlock(const std::string& name) const 
{
  // Iterate through all of the child nodes that have this name

  const DOMNode* found_node = d_node->getNextSibling();

  while(found_node != 0){
    char *s = XMLString::transcode(found_node->getNodeName());
    std::string c_name(s);
    delete[] s;

    if (c_name == name) {
      break;
    }

    //const DOMNode* tmp = findNode(name,child);
    found_node = found_node->getNextSibling();

  }
   
  if (found_node == NULL) {
     return 0;
  }
  else {
     return scinew ProblemSpec(found_node, d_write);
  }
}


std::string ProblemSpec::getNodeName() const
{

  //DOMString node_name = d_node->getNodeName();
  char *s = XMLString::transcode(d_node->getNodeName());
  std::string name(s);
  delete[] s;

  return name;
}


ProblemSpecP ProblemSpec::get(const std::string& name, double &value)
{
  ProblemSpecP ps = this;

  const DOMNode* found_node = findNode(name,d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char* s = XMLString::transcode(child->getNodeValue());
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
  const DOMNode* found_node = findNode(name,d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char* s = XMLString::transcode(child->getNodeValue());
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
  const DOMNode* found_node = findNode(name,d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char *s = XMLString::transcode(child->getNodeValue());
	std::string cmp(s);
	delete[] s;
	// Slurp up any spaces that were put in before or after the cmp string.
	istringstream result_stream(cmp);
        string nospace_cmp;
        result_stream >> nospace_cmp;
	if (nospace_cmp == "false") {
	   value = false;
	}
	else if  (nospace_cmp == "true") {
	  value = true;
	} else {
	  string error = name + "Must be either true or false";
	  throw ProblemSetupException(error);
	}
      }
    }
  }
          
  return ps;

}

ProblemSpecP ProblemSpec::get(const std::string& name, std::string &value)
{
  ProblemSpecP ps = this;
  const DOMNode* found_node = findNode(name,d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char *s = XMLString::transcode(child->getNodeValue());
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
  const DOMNode* found_node = findNode(name, d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char *s = XMLString::transcode(child->getNodeValue());
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
			      vector<double>& value)
{

  std::string string_value;
  ProblemSpecP ps = this;
  const DOMNode* found_node = findNode(name, d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char *s = XMLString::transcode(child->getNodeValue());
	string_value = std::string(s);
	delete[] s;

	istringstream in(string_value);
	char c,next;
	string result;
	while (!in.eof()) {
	  in >> c;
	  if (c == '[' || c == ',' || c == ' ' || c == ']')
	    continue;
	  next = in.peek();
	  result += c;
	  if (next == ',' ||  next == ' ' || next == ']') {
	    // turn the result into a number
	    double val = atof(result.c_str());
	    value.push_back(val);
	    result.erase();
	  }
	}
      }
    }
  }
          
  return ps;

}


ProblemSpecP ProblemSpec::get(const std::string& name, 
			      vector<int>& value)
{

  std::string string_value;
  ProblemSpecP ps = this;
  const DOMNode* found_node = findNode(name, d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char *s = XMLString::transcode(child->getNodeValue());
	string_value = std::string(s);
	delete[] s;

	istringstream in(string_value);
	char c,next;
	string result;
	while (!in.eof()) {
	  in >> c;
	  if (c == '[' || c == ',' || c == ' ' || c == ']')
	    continue;
	  next = in.peek();
	  result += c;
	  if (next == ',' ||  next == ' ' || next == ']') {
	    // turn the result into a number
	    int val = atoi(result.c_str());
	    value.push_back(val);
	    result.erase();
	  }
	}
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
  const DOMNode* found_node = findNode(name, d_node);
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
	 child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	//DOMString val = child->getNodeValue();
	char *s = XMLString::transcode(child->getNodeValue());
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
			  vector<double>& value)
{

  // Check if the prob_spec is NULL

 if (! this->get(name,value))
      throw ParameterNotFound(name);

}

void ProblemSpec::require(const std::string& name, 
			  vector<int>& value)
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
  DOMNode* attr_node;
  const DOMNode* found_node = findNode(name,d_node);
  std::cout << "node name = " << found_node->getNodeName() << std::endl;
  if (found_node == NULL) {
    ps = 0;
    return ps;
  }
  else {
    if (found_node->getNodeType() == DOMNode::ELEMENT_NODE) {
      // We have an element node and attributes
      DOMNamedNodeMap* attr = found_node->getAttributes();
      int num_attr = attr->getLength();
      if (num_attr >= 1)
	attr_node = attr->item(0);
      else 
	return ps = 0;
      if (attr_node->getNodeType() == DOMNode::ATTRIBUTE_NODE) {
	//DOMString val = attr_node->getNodeValue();
	char *s = XMLString::transcode(attr_node->getNodeValue());
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

void ProblemSpec::getAttributes(map<string,string>& attributes)
{

  DOMNamedNodeMap* attr = d_node->getAttributes();
  int num_attr = attr->getLength();

  for (int i = 0; i<num_attr; i++) {
    string name(XMLString::transcode(attr->item(i)->getNodeName()));
    string value(XMLString::transcode(attr->item(i)->getNodeValue()));
		
    attributes[name]=value;
  }

}

bool ProblemSpec::getAttribute(const string& attribute, string& result)
{

  DOMNamedNodeMap* attr = d_node->getAttributes();
  //DOMString search_name(attribute.c_str());

  const DOMNode* n = attr->getNamedItem(XMLString::transcode(attribute.c_str()));
  if(n == 0)
     return false;
  //DOMString val = n.getNodeValue();
  char* s = XMLString::transcode(n->getNodeValue());
  result=s;
  delete[] s;
  return true;
}

const Uintah::TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}
