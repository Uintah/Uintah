#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <sgi_stl_warnings_on.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMException.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

using namespace Uintah;
using namespace SCIRun;

using namespace std;

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
  if (d_node == 0)
    return 0;
  const DOMNode *child = d_node->getFirstChild();
  while (child != 0) {
    const char* s = XMLString::transcode(child->getNodeName());
    string child_name(s);
    delete [] s;
    if (name == child_name) {
      return scinew ProblemSpec(child, d_write);
    }
    child = child->getNextSibling();
  }
  return 0;
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
  // Iterate through all of the child nodes of the next node
  // until one is found that has this name

  const DOMNode* found_node = d_node->getNextSibling();

  while(found_node != 0){
    const char *s = XMLString::transcode(found_node->getNodeName());
    std::string c_name(s);
    delete [] s;

    if (c_name == name) {
      break;
    }

    found_node = found_node->getNextSibling();
  }
  if (found_node == NULL) {
     return 0;
  }
  else {
     return scinew ProblemSpec(found_node, d_write);
  }
}

ProblemSpecP ProblemSpec::findTextBlock()
{
   for (DOMNode* child = d_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
     if (child->getNodeType() == DOMNode::TEXT_NODE) {
       return scinew ProblemSpec(child, d_write);
      }
   }

   return NULL;
}

std::string ProblemSpec::getNodeName() const
{

  char*s = XMLString::transcode(d_node->getNodeName());
  string name(s);
  delete [] s;
  return name;
}

short ProblemSpec::getNodeType() {
  return d_node->getNodeType();
}

ProblemSpecP ProblemSpec::importNode(ProblemSpecP src, bool deep) {
  DOMNode* d = d_node->getOwnerDocument()->importNode(src->getNode(), deep);
  if (d)
    return scinew ProblemSpec(d, d_write);
  else
    return 0;
}

ProblemSpecP ProblemSpec::replaceChild(ProblemSpecP toreplace, 
					ProblemSpecP replaced) {
  DOMNode* d = d_node->replaceChild(toreplace->getNode(), replaced->getNode());
  if (d)
    return scinew ProblemSpec(d, d_write);
  else
    return 0;
}

ProblemSpecP ProblemSpec::removeChild(ProblemSpecP child) {
  DOMNode* d = d_node->removeChild(child->getNode());
  if (d)
    return scinew ProblemSpec(d, d_write);
  else 
    return 0;
}

//______________________________________________________________________
//
void checkForInputError(const std::string& stringValue, 
                        const std::string& Int_or_float)
{
  //__________________________________
  //  Make sure stringValue only contains valid characters
  if ("Int_or_float" != "int") {
    string validChars(" -+.0123456789eE");
    std::string::size_type  pos = stringValue.find_first_not_of(validChars);
    if (pos != string::npos){
      ostringstream warn;
      warn << "Input file error: I found ("<< stringValue[pos]
           << ") inside of "<< stringValue<< " at position "<< pos
           << "\nIf this is a valid number tell me --Todd "<<endl;
      throw ProblemSetupException(warn.str());
    }
    //__________________________________
    // check for two or more "."
    std::string::size_type p1 = stringValue.find_first_of(".");    
    std::string::size_type p2 = stringValue.find_last_of(".");     
    if (p1 != p2){
      ostringstream warn;
      warn << "Input file error: I found two (..) "
           << "inside of "<< stringValue
           << "\nIf this is a valid number tell me --Todd "<<endl;
      throw ProblemSetupException(warn.str());
    }
  }  
  if (Int_or_float == "int")  {
    string validChars(" -0123456789");
    std::string::size_type  pos = stringValue.find_first_not_of(validChars);
    if (pos != string::npos){
      ostringstream warn;
      warn << "Input file error Integer Number: I found ("<< stringValue[pos]
           << ") inside of "<< stringValue<< " at position "<< pos
           << "\nIf this is a valid number tell me --Todd "<<endl;
      throw ProblemSetupException(warn.str());
    }
  }
} 

ProblemSpecP ProblemSpec::get(const std::string& name, double &value)
{
  ProblemSpecP ps = this;

  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
        const char* s = XMLString::transcode(child->getNodeValue());
        string stringValue(s);
	delete [] s;
        checkForInputError(stringValue,"double"); 
        value = atof(stringValue.c_str());
      }
    }
  }
          
  return ps;
}

ProblemSpecP ProblemSpec::get(const std::string& name, int &value)
{
  ProblemSpecP ps = this;
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char* s = XMLString::transcode(child->getNodeValue());
	string stringValue(s);
	delete [] s;
	checkForInputError(stringValue,"int");
	value = atoi(stringValue.c_str());
      }
    }
  }
          
  return ps;

}

ProblemSpecP ProblemSpec::get(const std::string& name, long &value)
{
  ProblemSpecP ps = this;
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char* s = XMLString::transcode(child->getNodeValue());
	string stringValue(s);
	delete [] s;
	checkForInputError(stringValue,"int");
	value = atoi(stringValue.c_str());
      }
    }
  }
          
  return ps;

}

ProblemSpecP ProblemSpec::get(const std::string& name, bool &value)
{
  ProblemSpecP ps = this;
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char *s = XMLString::transcode(child->getNodeValue());
	std::string cmp(s);
	delete [] s;
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
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
        const char *s = XMLString::transcode(child->getNodeValue());
        //__________________________________
        // This little bit of magic removes all spaces
        std::string tmp(s);
	delete [] s;
        istringstream value_tmp(tmp);
        value_tmp >> value; 
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
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char *s = XMLString::transcode(child->getNodeValue());
	string_value = std::string(s);
	delete [] s;
	// Parse out the [num,num,num]
	// Now pull apart the string_value
	std::string::size_type i1 = string_value.find("[");
	std::string::size_type i2 = string_value.find_first_of(",");
	std::string::size_type i3 = string_value.find_last_of(",");
	std::string::size_type i4 = string_value.find("]");
	
	std::string x_val(string_value,i1+1,i2-i1-1);
	std::string y_val(string_value,i2+1,i3-i2-1);
	std::string z_val(string_value,i3+1,i4-i3-1);
       
	checkForInputError(x_val, "double"); 
	checkForInputError(y_val, "double");
	checkForInputError(z_val, "double");

	value.x(atof(x_val.c_str()));
	value.y(atof(y_val.c_str()));
	value.z(atof(z_val.c_str()));	
      }
    }
  }
          
  return ps;

}

// value should probably be empty before calling this...
ProblemSpecP ProblemSpec::get(const std::string& name, 
                           vector<double>& value)
{

  std::string string_value;
  ProblemSpecP ps = this;
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char *s = XMLString::transcode(child->getNodeValue());
	string_value = std::string(s);
	delete [] s;

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
           checkForInputError(result, "double"); 
           
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

// value should probably be empty before calling this...
ProblemSpecP ProblemSpec::get(const std::string& name, 
                           vector<int>& value)
{

  std::string string_value;
  ProblemSpecP ps = this;
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char *s = XMLString::transcode(child->getNodeValue());
	string_value = std::string(s);
	delete [] s;

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

           checkForInputError(result, "int"); 
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
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    for (DOMNode* child = found_node->getFirstChild(); child != 0;
        child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	const char *s = XMLString::transcode(child->getNodeValue());
	string_value = std::string(s);
	delete [] s;

	// Parse out the [num,num,num]
	// Now pull apart the string_value
	std::string::size_type i1 = string_value.find("[");
	std::string::size_type i2 = string_value.find_first_of(",");
	std::string::size_type i3 = string_value.find_last_of(",");
	std::string::size_type i4 = string_value.find("]");
	
	std::string x_val(string_value,i1+1,i2-i1-1);
	std::string y_val(string_value,i2+1,i3-i2-1);
	std::string z_val(string_value,i3+1,i4-i3-1);

	checkForInputError(x_val, "int");     
	checkForInputError(y_val, "int");     
	checkForInputError(z_val, "int");     
                
	value.x(atoi(x_val.c_str()));
	value.y(atoi(y_val.c_str()));
	value.z(atoi(z_val.c_str()));	
      }
    }
  }
          
  return ps;

}

bool ProblemSpec::get(int &value)
{
   for (DOMNode *child = d_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = XMLString::transcode(child->getNodeValue());
	 value = atoi(s);
	 delete [] s;
	 return true;
      }
   }
   return false;
}

bool ProblemSpec::get(long &value)
{
   for (DOMNode *child = d_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = XMLString::transcode(child->getNodeValue());
	 value = atoi(s);
	 delete [] s;
	 return true;
      }
   }
   return false;
}


ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 double& value, double defaultVal) 
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 int& value, int defaultVal)
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value=defaultVal;
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 bool& value, bool defaultVal)
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value=defaultVal;
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 std::string& value, 
					 const std::string& defaultVal)
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }


  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 IntVector& value, 
					 const IntVector& defaultVal)
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 Vector& value, 
					 const Vector& defaultVal)
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 Point& value, 
					 const Point& defaultVal)
{
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;
    value = defaultVal;
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 vector<double>& value, 
					 const vector<double>& defaultVal)
{
  value.clear();
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    //create DOMNode to add to the tree
    appendElement(name.c_str(), defaultVal);

    // set default values
    ps = this;

    value.clear();
    int size = static_cast<int>(defaultVal.size());
    for (int i = 0; i < size; i++)
      value.push_back(defaultVal[i]);
  }

  return ps;
}
ProblemSpecP ProblemSpec::getWithDefault(const std::string& name, 
					 vector<int>& value, 
					 const vector<int>& defaultVal)
{
  value.clear();
  ProblemSpecP ps = get(name, value);
  if (ps == 0) {

    // add DOMNode to the tree
    appendElement(name.c_str(), defaultVal);
    // set default values
    ps = this;
    value.clear();
    int size = static_cast<int>(defaultVal.size());
    for (int i = 0; i < size; i++)
      value.push_back(defaultVal[i]);
  }

  return ps;
}

void ProblemSpec::appendElement(const char* name, const std::string& value,
                                bool trail /*=0*/, int tabs /*=1*/) 
{
  ostringstream ostr;
  ostr.clear();
  ostr << "\n";
  for (int i = 0; i < tabs; i++)
    ostr << "\t";

   XMLCh* str = XMLString::transcode(ostr.str().c_str());

   DOMText *leader = d_node->getOwnerDocument()->createTextNode(str);
   delete [] str;
   d_node->appendChild(leader);

   str = XMLString::transcode(name);
   DOMElement *newElem = d_node->getOwnerDocument()->createElement(str);
   delete [] str;
   d_node->appendChild(newElem);

   str = XMLString::transcode(value.c_str());
   DOMText *newVal = d_node->getOwnerDocument()->createTextNode(str);
   delete [] str;
   newElem->appendChild(newVal);

   if (trail) {
     str = XMLString::transcode("\n");
     DOMText *trailer = d_node->getOwnerDocument()->createTextNode(str);
     delete [] str;
     d_node->appendChild(trailer);
   }
}

//basically to make sure correct overloaded function is called
void ProblemSpec::appendElement(const char* name, const char* value,
                                bool trail /*=0*/, int tabs /*=1*/) {
  appendElement(name, string(value), trail, tabs);
}


void ProblemSpec::appendElement(const char* name, int value,
                                bool trail /*=0*/, int tabs /*=1*/) 
{
   ostringstream val;
   val << value;
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, long value,
                                bool trail /*=0*/, int tabs /*=1*/) 
{
   ostringstream val;
   val << value;
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, const IntVector& value,
                                bool trail /*=0*/, int tabs /*=1*/) 
{
   ostringstream val;
   val << '[' << value.x() << ", " << value.y() << ", " << value.z() << ']';
   appendElement(name, val.str(), trail, tabs);
}

void ProblemSpec::appendElement(const char* name, const Point& value,
                                bool trail /*=0*/, int tabs /*=1*/) 
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, const Vector& value,
                                bool trail /*=0*/, int tabs /*=1*/)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, double value,
                                bool trail /*=0*/, int tabs /*=1*/)
{
   ostringstream val;
   val << setprecision(17) << value;
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, const vector<double>& value,
                                bool trail /*=0*/, int tabs /*=1*/)
{
   ostringstream val;
   val << '[';
   for (unsigned int i = 0; i < value.size(); i++) {
     val << setprecision(17) << value[i];
     if (i !=  value.size()-1)
       val << ',';
     
   }
   val << ']';
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, const vector<int>& value,
                                bool trail /*=0*/, int tabs /*=1*/)
{
   ostringstream val;
   val << '[';
   for (unsigned int i = 0; i < value.size(); i++) {
     val << setprecision(17) << value[i];
     if (i !=  value.size()-1)
       val << ',';
     
   }
   val << ']';
   appendElement(name, val.str(), trail, tabs);

}

void ProblemSpec::appendElement(const char* name, bool value,
                                bool trail /*=0*/, int tabs /*=1*/)
{
  if (value)
    appendElement(name, string("true"), trail, tabs);
  else
    appendElement(name, string("false"), trail, tabs);
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

void ProblemSpec::require(const std::string& name, long& value)
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
  ProblemSpecP node = findBlock(name);
  if (node == 0) {
    ps = 0;
    return ps;
  }
  else {
    DOMNode* found_node = node->d_node;
    std::cout << "node name = " << found_node->getNodeName() << std::endl;
    if (found_node->getNodeType() == DOMNode::ELEMENT_NODE) {
      // We have an element node and attributes
      DOMNamedNodeMap* attr = found_node->getAttributes();
      unsigned long num_attr = attr->getLength();
      if (num_attr >= 1)
	attr_node = attr->item(0);
      else 
	return ps = 0;
      if (attr_node->getNodeType() == DOMNode::ATTRIBUTE_NODE) {
	const char *s = XMLString::transcode(attr_node->getNodeValue());
	value = std::string(s);
	delete [] s;
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
  unsigned long num_attr = attr->getLength();

  for (unsigned long i = 0; i<num_attr; i++) {
    const char* attrName = XMLString::transcode(attr->item(i)->getNodeName());
    string name(attrName);
    delete [] attrName;

    const char* attrValue = XMLString::transcode(attr->item(i)->getNodeValue());
    string value(attrValue);
    delete [] attrValue;

    attributes[name]=value;
  }

}

bool ProblemSpec::getAttribute(const string& attribute, string& result)
{

  DOMNamedNodeMap* attr = d_node->getAttributes();

  const XMLCh* attrName = XMLString::transcode(attribute.c_str());
  const DOMNode* n = attr->getNamedItem(attrName);

  if(n == 0)
     return false;
  const char* s = XMLString::transcode(n->getNodeValue());
  result=s;
  delete [] s;
  delete [] attrName;
  return true;
}

void ProblemSpec::setAttribute(const std::string& name, 
				const std::string& value)
{
  XMLCh* attrName = XMLString::transcode(name.c_str());
  XMLCh* attrValue = XMLString::transcode(value.c_str());

  DOMElement * casted;
#if !defined(_AIX)
  casted = dynamic_cast<DOMElement*>(d_node);
  if( !casted ) {
    cerr << "Dynamic_cast of ProblemSpec.cc: d_node failed\n";
  }
#else
  casted = static_cast<DOMElement*>(d_node);
#endif

  casted->setAttribute(attrName, attrValue);

  delete [] attrName;
  delete [] attrValue;
}


ProblemSpecP ProblemSpec::getFirstChild() {
  DOMNode* d = d_node->getFirstChild();
  if (d)
    return scinew ProblemSpec(d, d_write);
  else
    return 0;
}

ProblemSpecP ProblemSpec::getNextSibling() {
  DOMNode* d = d_node->getNextSibling();
  if (d)
    return scinew ProblemSpec(d, d_write);
  else
    return 0;
}

string ProblemSpec::getNodeValue() {
  const char* value = XMLString::transcode(d_node->getNodeValue());
  string ret(value);
  delete [] value;
  return ret;
}

ProblemSpecP ProblemSpec::createElement(char* str) {
  XMLCh* newstr = XMLString::transcode(str);
  DOMNode* ret = d_node->getOwnerDocument()->createElement(newstr);
  delete [] newstr;
  return scinew ProblemSpec(ret, d_write);
}
void ProblemSpec::appendText(const char* str) {
  XMLCh* newstr = XMLString::transcode(str);
  d_node->appendChild(d_node->getOwnerDocument()->createTextNode(newstr));
  delete [] newstr;
}

// append element with associated string
// preceded by \n with tabs tabs (default 0), and followed by a newline
ProblemSpecP ProblemSpec::appendChild(const char *str, int tabs) {
  ostringstream ostr;
  ostr.clear();
  for (int i = 0; i < tabs; i++)
    ostr << "\t";
  appendText(ostr.str().c_str());
  
  XMLCh* newstr = XMLString::transcode(str);
  DOMNode* elt = d_node->getOwnerDocument()->createElement(newstr);

  delete [] newstr;
  newstr = XMLString::transcode("\n");
  
  // add trailing newline
  d_node->appendChild(elt);
  d_node->appendChild(d_node->getOwnerDocument()->createTextNode(newstr));
  delete [] newstr;
  return scinew ProblemSpec(elt, d_write);
}

void ProblemSpec::appendChild(ProblemSpecP pspec) {
  d_node->appendChild(pspec->d_node);
}

void ProblemSpec::addStylesheet(char* type, char* name) {
   ASSERT((strcmp(type, "css") == 0) || strcmp(type, "xsl") == 0);

   XMLCh* str1 = XMLString::transcode("xml-stylesheet");

   ostringstream str;
   str << " type=\"text/" << type << "\" href=\"" << name << "\"";
   XMLCh* str2 = XMLString::transcode(str.str().c_str());

   DOMProcessingInstruction* pi = d_node->getOwnerDocument()->createProcessingInstruction(str1,str2);

   delete [] str1;
   delete [] str2;

   d_node->getOwnerDocument()->insertBefore((DOMNode*)pi, (DOMNode*)(d_node->getOwnerDocument()->getDocumentElement()));

}

// filename is a default param (NULL)
//   call with no parameters or NULL to output
//   to stdout
void ProblemSpec::output(char* filename) const {
  XMLCh* tempStr = XMLString::transcode("LS");
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(tempStr);
  delete [] tempStr;

  DOMWriter* writer = ((DOMImplementationLS*)impl)->createDOMWriter();
  if (writer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  XMLFormatTarget *target;
  if (filename == NULL)
    target = new StdOutFormatTarget();
  else
    target = new LocalFileFormatTarget(filename);

  try {
    // do the serialization through DOMWriter::writeNode();
    writer->writeNode(target, *(d_node->getOwnerDocument()));
  }
  catch (const XMLException& toCatch) {
    char* message = XMLString::transcode(toCatch.getMessage());
    string ex(message);
    delete [] message;
    throw ProblemSetupException(ex);
  }
  catch (const DOMException& toCatch) {
    char* message = XMLString::transcode(toCatch.msg);
    string ex(message);
    delete [] message;
    throw ProblemSetupException(ex);
  }
}

void ProblemSpec::releaseDocument() {
  d_node->getOwnerDocument()->release();
}

const Uintah::TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}

// static
ProblemSpecP ProblemSpec::createDocument(const std::string& name) {
  const XMLCh* str = XMLString::transcode(name.c_str());
  const XMLCh* implstr = XMLString::transcode("LS");
  DOMImplementation* impl = DOMImplementationRegistry::getDOMImplementation(implstr);
  DOMDocument* doc = impl->createDocument(0,str,0);
  delete [] str;
  delete [] implstr;

  return scinew ProblemSpec(doc->getDocumentElement());

}
ostream& operator<<(ostream& out, ProblemSpecP pspec) {
  out << pspec->getNode()->getOwnerDocument();
  return out;
}

void outputContent(ostream& target, const char *chars /**to_write*/)
{
  //const char* chars = strdup(to_char_ptr(to_write));
  for (unsigned int index = 0; index < strlen(chars); index++) {
    switch (chars[index]) {
    case chAmpersand :
      target << "&amp;";
      break;

    case chOpenAngle :
      target << "&lt;";
      break;

    case chCloseAngle:
      target << "&gt;";
      break;

    case chDoubleQuote :
      target << "&quot;";
      break;

    default:
      // If it is none of the special characters, print it as such
      target << chars[index];
      break;
    }
  }
  delete[] chars;
}


std::ostream& operator<<(std::ostream& target, const DOMNode* toWrite) {
  // Get the name and value out for convenience
  const char *nodeName = XMLString::transcode(toWrite->getNodeName());
  const char *nodeValue = XMLString::transcode(toWrite->getNodeValue());

  // nodeValue will be sometimes be deleted in outputContent, but
  // will not always call outputContent
  bool valueDeleted = false;

  switch (toWrite->getNodeType()) {
  case DOMNode::TEXT_NODE:
    {
      outputContent(target, nodeValue);
      valueDeleted = true;
      break;
    }

  case DOMNode::PROCESSING_INSTRUCTION_NODE :
    {
      target  << "<?"
	      << nodeName
	      << ' '
	      << nodeValue
	      << "?>";
      break;
    }

  case DOMNode::DOCUMENT_NODE :
    {
      // Bug here:  we need to find a way to get the encoding name
      //   for the default code page on the system where the
      //   program is running, and plug that in for the encoding
      //   name.
      //MLCh *enc_name = XMLPlatformUtils::fgTransService->getEncodingName();
      target << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";

      DOMNode *brother = toWrite->getNextSibling();
      while(brother != 0)
        {
          target << brother << endl;
          brother = brother->getNextSibling();
        }

      DOMNode *child = toWrite->getFirstChild();
      while(child != 0)
        {
          target << child << endl;
          child = child->getNextSibling();
        }

      break;
    }

  case DOMNode::ELEMENT_NODE :
    {
      // Output the element start tag.
      target << '<' << nodeName;

      // Output any attributes on this element
      DOMNamedNodeMap *attributes = toWrite->getAttributes();
      int attrCount = static_cast<int>(attributes->getLength());
      for (int i = 0; i < attrCount; i++) {
	DOMNode  *attribute = attributes->item(i);
	char* attrName = XMLString::transcode(attribute->getNodeName());
	target  << ' ' << attrName
		<< " = \"";
	//  Note that "<" must be escaped in attribute values.
	outputContent(target, XMLString::transcode(attribute->getNodeValue(\
									   )));
	target << '"';
	delete [] attrName;
      }

      //  Test for the presence of children, which includes both
      //  text content and nested elements.
      DOMNode *child = toWrite->getFirstChild();
      if (child != 0) {
	// There are children. Close start-tag, and output children.
	target << ">";
	while(child != 0) {
	  target << child;
	  child = child->getNextSibling();
	}

	// Done with children.  Output the end tag.
	target << "</" << nodeName << ">";
      } else {
	//  There were no children.  Output the short form close of the
	//  element start tag, making it an empty-element tag.
	target << "/>";
      }
      break;
    }

  case DOMNode::ENTITY_REFERENCE_NODE:
    {
      DOMNode *child;
      for (child = toWrite->getFirstChild(); child != 0;
	   child = child->getNextSibling())
	target << child;
      break;
    }

  case DOMNode::CDATA_SECTION_NODE:
    {
      target << "<![CDATA[" << nodeValue << "]]>";
      break;
    }

  case DOMNode::COMMENT_NODE:
    {
      target << "<!--" << nodeValue << "-->";
      break;
    }

  default:
    cerr << "Unrecognized node type = "
	 << (long)toWrite->getNodeType() << endl;
  }

  delete [] nodeName;
  if (!valueDeleted)
    delete [] nodeValue;
  return target;
}

std::ostream& operator<<(std::ostream& target, const DOMText* toWrite) {
  const char *p = XMLString::transcode(toWrite->getData());
  target << p;
  delete [] p;
  return target;

}
