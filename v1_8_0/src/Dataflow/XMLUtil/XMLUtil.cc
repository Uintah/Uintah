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


#include <Core/Util/notset.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <strings.h>
#include <stdio.h>
using namespace std;

namespace SCIRun {

static void postMessage(const string& errmsg)
{
    cerr << errmsg << '\n';
}

const DOMNode* findNode(const std::string &name, const DOMNode *node)
{
  // Convert string name to a DOMText;
  if (node == 0)
    return 0;
  // const XMLCh *search_name = to_xml_ch_ptr(name.c_str());
  // Do the child nodes now - convert to char*
  const DOMNode *child = node->getFirstChild();
  while (child != 0) {
    const char* child_name = to_char_ptr(child->getNodeName());
    if (child_name && strcmp(name.c_str(), child_name) == 0) {
      return child;
    }
    child = child->getNextSibling();
  }
  return 0;
}

const DOMNode* findNextNode(const std::string& name, const DOMNode* node)
{
  // Iterate through all of the child nodes that have this name
  DOMNode* found_node = node->getNextSibling();

  while(found_node != 0){
    const char* found_node_name = to_char_ptr(found_node->getNodeName());
    if (strcmp(name.c_str(), found_node_name) == 0 ) {
      break;
    }
    found_node = found_node->getNextSibling();
  }
  return found_node;
}


const DOMNode* findTextNode(const DOMNode* node)
{
   for (DOMNode* child = node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 return child;
      }
   }

   //not sure what to do here...
  DOMNode* unknown = NULL;
  return unknown;   
}

// string toString(const XMLCh* const str)
// {
//     char* s = XMLString::transcode(str);
//     if(!s)
//        return "";
//     string ret = string(s);
//     delete[] s;
//     return ret;
// }

// string toString(const DOMText* str)
// {
//     char* s = str.transcode();
//     if(!s)
//        return "";
//     string ret = string(s);
//     delete[] s;
//     return ret;
// }

// ---------------------------------------------------------------------------
//  ostream << DOMNode   
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOMNode* toWrite)
{
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
	 int attrCount = attributes->getLength();
	 for (int i = 0; i < attrCount; i++) {
	    DOMNode  *attribute = attributes->item(i);
	    char* attrName = XMLString::transcode(attribute->getNodeName());
	    target  << ' ' << attrName
		    << " = \"";
	    //  Note that "<" must be escaped in attribute values.
	    outputContent(target, XMLString::transcode(attribute->getNodeValue()));
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


// ---------------------------------------------------------------------------
//  outputContent  - Write document content from a DOMCh* to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
// ---------------------------------------------------------------------------
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


// ---------------------------------------------------------------------------
//  ostream << DOMText    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOMText* s)
{
   const char *p = to_char_ptr(s->getData());
   target << p;
   return target;
}

void appendElement(DOMElement* root, const DOMText* name,
		   const std::string& value)
{
   DOMText *leader = 
     root->getOwnerDocument()->createTextNode(to_xml_ch_ptr("\n\t"));
   root->appendChild(leader);
   DOMElement *newElem = 
     root->getOwnerDocument()->createElement(name->getData());
   root->appendChild(newElem);
   DOMText *newVal = 
     root->getOwnerDocument()->createTextNode(to_xml_ch_ptr(value.c_str()));
   newElem->appendChild(newVal);
   DOMText *trailer = 
     root->getOwnerDocument()->createTextNode(to_xml_ch_ptr("\n"));
   root->appendChild(trailer);
}
      
void appendElement(DOMElement* root, const DOMText* name,
		   int value)
{
   ostringstream val;
   val << value;
   appendElement(root, name, val.str());
}
      
void appendElement(DOMElement* root, const DOMText* name,
		   const IntVector& value)
{
   ostringstream val;
   val << '[' << value.x() << ", " << value.y() << ", " << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOMElement* root, const DOMText* name,
		   const Point& value)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOMElement* root, const DOMText* name,
		   const Vector& value)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOMElement* root, const DOMText* name,
		   long value)
{
   ostringstream val;
   val << value;
   appendElement(root, name, val.str());
}
      
void appendElement(DOMElement* root, const DOMText* name,
		   double value)
{
   ostringstream val;
   val << setprecision(17) << value;
   appendElement(root, name, val.str());
}
      
bool get(const DOMNode* node, int &value)
{
   for (DOMNode *child = node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = to_char_ptr(child->getNodeValue());
	 value = atoi(s);
	 return true;
      }
   }
   return false;
}

bool 
get(const DOMNode* node, const std::string& name, int &value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
     return false;
   return get(found_node, value);
}

bool get(const DOMNode* node, long &value)
{
   for (DOMNode *child = node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char *s = to_char_ptr(child->getNodeValue());
	 value = atoi(s);
	 return true;
      }
   }
   return false;
}

bool 
get(const DOMNode* node, const std::string& name, long &value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
      return false;
   return get(found_node, value);
}

bool 
get(const DOMNode* node, const std::string& name, double &value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
      return false;
   for (DOMNode *child = found_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = to_char_ptr(child->getNodeValue());
	 value = atof(s);
	 return true;
      }
   }
   return false;
}

bool 
get(const DOMNode* node, const std::string& name, std::string &value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
      return false;
   for (DOMNode *child = found_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = to_char_ptr(child->getNodeValue());
	 value = std::string(s);
	 return true;
      }
   }
   return false;
}

bool 
get(const DOMNode* node, const std::string& name, Vector& value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
      return false;
   for (DOMNode *child = found_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = to_char_ptr(child->getNodeValue());
	 string string_value = std::string(s);
	 
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
	 return true;
      }
   }
   return false;
}

bool 
get(const DOMNode* node, const std::string& name, Point& value)
{
   Vector v;
   bool status = get(node, name, v);
   value=Point(v);
   return status;
}

bool 
get(const DOMNode* node, const std::string& name, IntVector &value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
      return false;
   for (DOMNode *child = found_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = to_char_ptr(child->getNodeValue());
	 string string_value = std::string(s);
	 
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
	 return true;
      }
   }
   return false;
}

bool 
get(const DOMNode* node, const std::string& name, bool &value)
{
   const DOMNode *found_node = findNode(name, node);
   if(!found_node)
      return false;
   for (DOMNode *child = found_node->getFirstChild(); child != 0;
	child = child->getNextSibling()) {
      if (child->getNodeType() == DOMNode::TEXT_NODE) {
	 const char* s = to_char_ptr(child->getNodeValue());
	 std::string cmp(s);
	 
	 if (cmp == "false")
	    value = false;
	 else if (cmp == "true")
	    value = true;
	
	 return true;
      }
   }
   return false;
}

char* getSerializedAttributes(DOMNode* d)
{
  char* str = 0;
  char* fullstr = new char[1];
  char* newstr = 0;

  fullstr[0]='\0';

  DOMNamedNodeMap *attr = d->getAttributes();
  int length = attr->getLength();
  int index = 0;
  // This can only be a DOMAttr*
  for(DOMAttr *n = (DOMAttr*)attr->item(index); index != length;
      n = (DOMAttr*)attr->item(++index)) {
    const string nn(to_char_ptr(n->getName()));
    const char* nv = to_char_ptr(n->getValue());
    str = new char[strlen(nn.c_str()) + strlen(nv) + 5];
    sprintf(str, " %s=\"%s\"", nn.c_str(), nv);

    int newlength = strlen(str)+strlen(fullstr);
    newstr = new char[newlength+1];
    newstr[0]='\0';
    sprintf(newstr,"%s%s",fullstr,str);
    delete[] fullstr;
    fullstr = newstr;
    newstr = 0;

    delete[] str;
    str = 0;
  }

  return fullstr;
}

char* getSerializedChildren(DOMNode* d)
{
  char* temp = 0;
  char* temp2 = 0;
  char* str = 0;
  char* fullstr = new char[1];
  char* newstr = 0;

  fullstr[0]='\0';
  for (DOMNode *n = d->getFirstChild(); n != 0; n = n->getNextSibling()) {

    if (n->getNodeType() == DOMNode::TEXT_NODE) {      
      DOMText *dt = (DOMText*)n;
      const char *nv = to_char_ptr(dt->getNodeValue());
      str = new char[strlen(nv) + 1];
      str[0]='\0';
      sprintf(str, "%s", nv);
    } else if (n->getNodeType() == DOMNode::ELEMENT_NODE){
      DOMElement *de = (DOMElement*)n;
      string tname(to_char_ptr(de->getTagName()));
      temp = getSerializedAttributes(n);
      temp2 = getSerializedChildren(n);
      str = new char[2 * strlen(tname.c_str()) + strlen(temp) + strlen(temp2) + 6];
      str[0]='\0';
      sprintf(str, "<%s%s>%s</%s>", tname.c_str(), temp, temp2, tname.c_str());
      delete[] temp;
      delete[] temp2;
    } else {
      ASSERTFAIL("unexpected node type, in XMLUtil.cc");
    }
    int newlength = strlen(str) + strlen(fullstr);
    newstr = new char[newlength+1];
    newstr[0]='\0';
    
    sprintf(newstr,"%s%s",fullstr,str);
    delete[] fullstr;
    fullstr = newstr;
    newstr = 0;
    
    delete[] str;
    str = 0;
  }
  return fullstr;
}

string xmlto_string(const DOMText* str)
{
  const char* s = to_char_ptr(str->getData());
  string ret = string(s);
  return ret;
}

string xmlto_string(const XMLCh* const str)
{
  const char* s = to_char_ptr(str);
  string ret = string(s);
  return ret;
}

void invalidNode(const DOMNode* n, const string& filename)
{
  if(n->getNodeType() == DOMNode::COMMENT_NODE)
      return;
  if(n->getNodeType() == DOMNode::TEXT_NODE){
    const char* str = to_char_ptr(n->getNodeValue());
    bool allwhite=true;
    for(const char* p = str; *p != 0; p++){
      if(!isspace(*p))
	allwhite=false;
      }
    if(!allwhite){
      postMessage(string("Extraneous text: ") + str + "after node: " + 
		  xmlto_string(n->getNodeName()) + "(in file " + filename + 
		  ")");
    }
    return;
  }
  postMessage(string("Do not understand node: ") + 
	      xmlto_string(n->getNodeName()) + "(in file " + filename + ")");
}

const XMLCh* 
findText(DOMNode* node)
{
  for(DOMNode *n = node->getFirstChild();n != 0; n = n->getNextSibling()){
    if(n->getNodeType() == DOMNode::TEXT_NODE)
      return n->getNodeValue();
  }
  return 0;
}

} // End namespace SCIRun
