/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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



#include <Core/Util/notset.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#ifndef _WIN32
#include <strings.h>
#endif
#include <stdio.h>
using namespace std;

namespace SCIRun {

 
xmlAttrPtr
get_attribute_by_name(const xmlNodePtr p, const char *name) 
{
  xmlAttr *cur = p->properties;
  
  while (cur != 0) {
    if (cur->type == XML_ATTRIBUTE_NODE && string_is(cur->name, name)) {
      return cur;
    }
    cur = cur->next;
  }
  return 0;
} 

bool
get_attributes(vector<xmlNodePtr> &attr, xmlNodePtr p) 
{
  attr.clear();
  xmlAttr *cur = p->properties;
  
  while (cur != 0) {
    if (cur->type == XML_ATTRIBUTE_NODE) 
    {
      attr.push_back(cur->children);
    }
    cur = cur->next;
  }
  return attr.size() > 0;
} 



// static void post_message(const string& errmsg)
// {
//     cerr << errmsg << '\n';
// }

// const DOMNode* findNode(const std::string &name, const DOMNode *node)
// {
//   // Convert string name to a DOMText;
//   if (node == 0)
//     return 0;
//   // const XMLCh *search_name = to_xml_ch_ptr(name.c_str());
//   // Do the child nodes now - convert to char*
//   const DOMNode *child = node->getFirstChild();
//   while (child != 0) {
//     const char* child_name = to_char_ptr(child->getNodeName());
//     if (child_name && strcmp(name.c_str(), child_name) == 0) {
//       return child;
//     }
//     child = child->getNextSibling();
//   }
//   return 0;
// }

// const DOMNode* findNextNode(const std::string& name, const DOMNode* node)
// {
//   // Iterate through all of the child nodes that have this name
//   DOMNode* found_node = node->getNextSibling();

//   while(found_node != 0){
//     const char* found_node_name = to_char_ptr(found_node->getNodeName());
//     if (strcmp(name.c_str(), found_node_name) == 0 ) {
//       break;
//     }
//     found_node = found_node->getNextSibling();
//   }
//   return found_node;
// }


// const DOMNode* findTextNode(const DOMNode* node)
// {
//    for (DOMNode* child = node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 return child;
//       }
//    }

//    //not sure what to do here...
//   DOMNode* unknown = NULL;
//   return unknown;   
// }

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
// ostream& operator<<(ostream& target, const DOMNode* toWrite)
// {
//    Get the name and value out for convenience
//    const char *nodeName = XMLString::transcode(toWrite->getNodeName());
//    const char *nodeValue = XMLString::transcode(toWrite->getNodeValue());

//    nodeValue will be sometimes be deleted in outputContent, but 
//    will not always call outputContent
//    bool valueDeleted = false;
   
//    switch (toWrite->getNodeType()) {
//    case DOMNode::TEXT_NODE:
//       {
// 	 outputContent(target, nodeValue);
// 	 valueDeleted = true;
// 	 break;
//       }
   
//    case DOMNode::PROCESSING_INSTRUCTION_NODE :
//       {
// 	 target  << "<?"
// 		 << nodeName
// 		 << ' '
// 		 << nodeValue
// 		 << "?>";
// 	 break;
//       }
   
//    case DOMNode::DOCUMENT_NODE :
//       {
// 	 Bug here:  we need to find a way to get the encoding name
// 	   for the default code page on the system where the
// 	   program is running, and plug that in for the encoding
// 	   name.  
// 	MLCh *enc_name = XMLPlatformUtils::fgTransService->getEncodingName();
// 	target << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
// 	DOMNode *child = toWrite->getFirstChild();
// 	while(child != 0)
// 	{
// 	  target << child << endl;
// 	  child = child->getNextSibling();
// 	}
	
// 	break;
//       }
   
//    case DOMNode::ELEMENT_NODE :
//       {
// 	 Output the element start tag.
// 	 target << '<' << nodeName;
	 
// 	 Output any attributes on this element
// 	 DOMNamedNodeMap *attributes = toWrite->getAttributes();
// 	 int attrCount = attributes->getLength();
// 	 for (int i = 0; i < attrCount; i++) {
// 	    DOMNode  *attribute = attributes->item(i);
// 	    char* attrName = XMLString::transcode(attribute->getNodeName());
// 	    target  << ' ' << attrName
// 		    << " = \"";
// 	     Note that "<" must be escaped in attribute values.
// 	    outputContent(target, XMLString::transcode(attribute->getNodeValue()));
// 	    target << '"';
// 	    delete [] attrName;
// 	 }
	 
// 	  Test for the presence of children, which includes both
// 	  text content and nested elements.
// 	 DOMNode *child = toWrite->getFirstChild();
// 	 if (child != 0) {
// 	    There are children. Close start-tag, and output children.
// 	    target << ">";
// 	    while(child != 0) {
// 	       target << child;
// 	       child = child->getNextSibling();
// 	    }

// 	    Done with children.  Output the end tag.
// 	    target << "</" << nodeName << ">";
// 	 } else {
// 	     There were no children.  Output the short form close of the
// 	     element start tag, making it an empty-element tag.
// 	    target << "/>";
// 	 }
// 	 break;
//       }
   
//    case DOMNode::ENTITY_REFERENCE_NODE:
//       {
// 	 DOMNode *child;
// 	 for (child = toWrite->getFirstChild(); child != 0; 
// 	      child = child->getNextSibling())
// 	    target << child;
// 	 break;
//       }
   
//    case DOMNode::CDATA_SECTION_NODE:
//       {
// 	 target << "<![CDATA[" << nodeValue << "]]>";
// 	 break;
//       }
   
//    case DOMNode::COMMENT_NODE:
//       {
// 	 target << "<!--" << nodeValue << "-->";
// 	 break;
//       }
   
//    default:
//       cerr << "Unrecognized node type = "
// 	   << (long)toWrite->getNodeType() << endl;
//    }

//    delete [] nodeName;
//    if (!valueDeleted)
//      delete [] nodeValue;
//    return target;
// }


// ---------------------------------------------------------------------------
//  outputContent  - Write document content from a DOMCh* to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
// ---------------------------------------------------------------------------
// void outputContent(ostream& target, const char *chars to_write)
// {
//   const char* chars = strdup(to_char_ptr(to_write));
//   for (unsigned int index = 0; index < strlen(chars); index++) {
//     switch (chars[index]) {
//     case chAmpersand :
//       target << "&amp;";
//       break;
      
//     case chOpenAngle :
//       target << "&lt;";
//       break;
	 
//     case chCloseAngle:
//       target << "&gt;";
//       break;
	 
//     case chDoubleQuote :
//       target << "&quot;";
//       break;
	 
//     default:
//       If it is none of the special characters, print it as such
//       target << chars[index];
//       break;
//     }
//   }
//   delete[] chars;
// }


// ---------------------------------------------------------------------------
//  ostream << DOMText    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
// ---------------------------------------------------------------------------
// ostream& operator<<(ostream& target, const DOMText* s)
// {
//    const char *p = to_char_ptr(s->getData());
//    target << p;
//    return target;
// }

// void appendElement(DOMElement* root, const DOMText* name,
// 		   const std::string& value)
// {
//    DOMText *leader = 
//      root->getOwnerDocument()->createTextNode(to_xml_ch_ptr("\n\t"));
//    root->appendChild(leader);
//    DOMElement *newElem = 
//      root->getOwnerDocument()->createElement(name->getData());
//    root->appendChild(newElem);
//    DOMText *newVal = 
//      root->getOwnerDocument()->createTextNode(to_xml_ch_ptr(value.c_str()));
//    newElem->appendChild(newVal);
//    DOMText *trailer = 
//      root->getOwnerDocument()->createTextNode(to_xml_ch_ptr("\n"));
//    root->appendChild(trailer);
// }
      
// void appendElement(DOMElement* root, const DOMText* name,
// 		   int value)
// {
//    ostringstream val;
//    val << value;
//    appendElement(root, name, val.str());
// }
      
// void appendElement(DOMElement* root, const DOMText* name,
// 		   const IntVector& value)
// {
//    ostringstream val;
//    val << '[' << value.x() << ", " << value.y() << ", " << value.z() << ']';
//    appendElement(root, name, val.str());
// }
      
// void appendElement(DOMElement* root, const DOMText* name,
// 		   const Point& value)
// {
//    ostringstream val;
//    val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
//    appendElement(root, name, val.str());
// }
      
// void appendElement(DOMElement* root, const DOMText* name,
// 		   const Vector& value)
// {
//    ostringstream val;
//    val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
//    appendElement(root, name, val.str());
// }
      
// void appendElement(DOMElement* root, const DOMText* name,
// 		   long value)
// {
//    ostringstream val;
//    val << value;
//    appendElement(root, name, val.str());
// }
      
// void appendElement(DOMElement* root, const DOMText* name,
// 		   double value)
// {
//    ostringstream val;
//    val << setprecision(17) << value;
//    appendElement(root, name, val.str());
// }
      
// bool get(const DOMNode* node, int &value)
// {
//    for (DOMNode *child = node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char* s = to_char_ptr(child->getNodeValue());
// 	 value = atoi(s);
// 	 return true;
//       }
//    }
//    return false;
// }

// bool 
// get(const DOMNode* node, const std::string& name, int &value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//      return false;
//    return get(found_node, value);
// }

// bool get(const DOMNode* node, long &value)
// {
//    for (DOMNode *child = node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char *s = to_char_ptr(child->getNodeValue());
// 	 value = atoi(s);
// 	 return true;
//       }
//    }
//    return false;
// }

// bool 
// get(const DOMNode* node, const std::string& name, long &value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//       return false;
//    return get(found_node, value);
// }

// bool 
// get(const DOMNode* node, const std::string& name, double &value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//       return false;
//    for (DOMNode *child = found_node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char* s = to_char_ptr(child->getNodeValue());
// 	 value = atof(s);
// 	 return true;
//       }
//    }
//    return false;
// }

// bool 
// get(const DOMNode* node, const std::string& name, std::string &value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//       return false;
//    for (DOMNode *child = found_node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char* s = to_char_ptr(child->getNodeValue());
// 	 value = std::string(s);
// 	 return true;
//       }
//    }
//    return false;
// }

// bool 
// get(const DOMNode* node, const std::string& name, Vector& value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//       return false;
//    for (DOMNode *child = found_node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char* s = to_char_ptr(child->getNodeValue());
// 	 string string_value = std::string(s);
	 
// 	 Parse out the [num,num,num]
// 	 Now pull apart the string_value
// 	 std::string::size_type i1 = string_value.find("[");
// 	 std::string::size_type i2 = string_value.find_first_of(",");
// 	 std::string::size_type i3 = string_value.find_last_of(",");
// 	 std::string::size_type i4 = string_value.find("]");
	
// 	 std::string x_val(string_value,i1+1,i2-i1-1);
// 	 std::string y_val(string_value,i2+1,i3-i2-1);
// 	 std::string z_val(string_value,i3+1,i4-i3-1);

// 	 value.x(atof(x_val.c_str()));
// 	 value.y(atof(y_val.c_str()));
// 	 value.z(atof(z_val.c_str()));	
// 	 return true;
//       }
//    }
//    return false;
// }

// bool 
// get(const DOMNode* node, const std::string& name, Point& value)
// {
//    Vector v;
//    bool status = get(node, name, v);
//    value=Point(v);
//    return status;
// }

// bool 
// get(const DOMNode* node, const std::string& name, IntVector &value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//       return false;
//    for (DOMNode *child = found_node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char* s = to_char_ptr(child->getNodeValue());
// 	 string string_value = std::string(s);
	 
// 	 Parse out the [num,num,num]
// 	 Now pull apart the string_value
// 	 std::string::size_type i1 = string_value.find("[");
// 	 std::string::size_type i2 = string_value.find_first_of(",");
// 	 std::string::size_type i3 = string_value.find_last_of(",");
// 	 std::string::size_type i4 = string_value.find("]");
	
// 	 std::string x_val(string_value,i1+1,i2-i1-1);
// 	 std::string y_val(string_value,i2+1,i3-i2-1);
// 	 std::string z_val(string_value,i3+1,i4-i3-1);
			
// 	 value.x(atoi(x_val.c_str()));
// 	 value.y(atoi(y_val.c_str()));
// 	 value.z(atoi(z_val.c_str()));	
// 	 return true;
//       }
//    }
//    return false;
// }

// bool 
// get(const DOMNode* node, const std::string& name, bool &value)
// {
//    const DOMNode *found_node = findNode(name, node);
//    if(!found_node)
//       return false;
//    for (DOMNode *child = found_node->getFirstChild(); child != 0;
// 	child = child->getNextSibling()) {
//       if (child->getNodeType() == DOMNode::TEXT_NODE) {
// 	 const char* s = to_char_ptr(child->getNodeValue());
// 	 std::string cmp(s);
	 
// 	 if (cmp == "false")
// 	    value = false;
// 	 else if (cmp == "true")
// 	    value = true;
	
// 	 return true;
//       }
//    }
//    return false;
// }

string get_serialized_attributes(xmlNode* d)
{
  string fullstr;

  vector<xmlNodePtr> attr;
  get_attributes(attr, d);
  vector<xmlNodePtr>::iterator iter = attr.begin();
  while(iter != attr.end()) {
    xmlNodePtr n = *iter++;
    ostringstream strm;
    strm << " " << fullstr << n->name << "=\"" << n->content << "\"";
    fullstr = strm.str();
  }

  return fullstr;
}

string get_serialized_children(xmlNode* d)
{
  string fullstr;

  for (xmlNode *n = d->children; n != 0; n = n->next) {
    string str;
    if (n->type == XML_TEXT_NODE) {
      str = string(to_char_ptr(n->content));
    } else if (n->type == XML_ELEMENT_NODE) {
      ostringstream strm;
      strm << "<" << n->name << get_serialized_attributes(n) << ">" 
	   << get_serialized_children(n) << "</" << n->name << ">";
      str = strm.str();
    } else {
      ASSERTFAIL("unexpected node type, in XMLUtil.cc");
    }
    fullstr = fullstr + str;
  }
  return fullstr;
}

// string xmlto_string(const DOMText* str)
// {
//   const char* s = to_char_ptr(str->getData());
//   string ret = string(s);
//   return ret;
// }

// string xmlto_string(const XMLCh* const str)
// {
//   const char* s = to_char_ptr(str);
//   string ret = string(s);
//   return ret;
// }

// void invalidNode(const DOMNode* n, const string& filename)
// {
//   if(n->getNodeType() == DOMNode::COMMENT_NODE)
//       return;
//   if(n->getNodeType() == DOMNode::TEXT_NODE){
//     const char* str = to_char_ptr(n->getNodeValue());
//     bool allwhite=true;
//     for(const char* p = str; *p != 0; p++){
//       if(!isspace(*p))
// 	allwhite=false;
//       }
//     if(!allwhite){
//       post_message(string("Extraneous text: ") + str + "after node: " + 
// 		  xmlto_string(n->getNodeName()) + "(in file " + filename + 
// 		  ")");
//     }
//     return;
//   }
//   post_message(string("Do not understand node: ") + 
// 	      xmlto_string(n->getNodeName()) + "(in file " + filename + ")");
// }

// const XMLCh* 
// findText(DOMNode* node)
// {
//   for(DOMNode *n = node->getFirstChild();n != 0; n = n->getNextSibling()){
//     if(n->getNodeType() == DOMNode::TEXT_NODE)
//       return n->getNodeValue();
//   }
//   return 0;
// }


namespace XMLUtil {

bool node_is_element(const xmlNodePtr p, const string &name) {
  return (p->type == XML_ELEMENT_NODE &&
	  name == xmlChar_to_char(p->name));
}

bool node_is_dtd(const xmlNodePtr p, const string &name) {
  return (p->type == XML_DTD_NODE &&
	  name == xmlChar_to_char(p->name));
}

bool node_is_comment(const xmlNodePtr p)
{
  //  ASSERT(0);
  return true;
}

bool maybe_get_att_as_int(const xmlNodePtr p, 
      			     const string &name, 
			     int &val)
{
  string str;
  return (maybe_get_att_as_string(p, name, str) &&
	  string_to_int(str, val));
}


bool
maybe_get_att_as_double(const xmlNodePtr p, 
			   const string &name, 
			   double &val)
{
  string str;
  return (maybe_get_att_as_string(p, name, str) &&
	  string_to_double(str, val));
}


bool
maybe_get_att_as_string(const xmlNodePtr p, 
			   const string &name, 
			   string &val)
{
  xmlAttrPtr attr = get_attribute_by_name(p, name.c_str());
  if (!attr)
    return false;
  val = xmlChar_to_string(attr->children->content);
  return true;
}

string node_att_as_string(const xmlNodePtr p, const string &name)
{
  xmlAttrPtr attr = get_attribute_by_name(p, name.c_str());
  if (!attr)
    throw "Attribute "+name+" does not exist!";
  return xmlChar_to_string(attr->children->content);
}

int node_att_as_int(const xmlNodePtr p, const string &name)
{
  int val = 0;
  string str = node_att_as_string(p, name);
  if (!string_to_int(str, val))
    throw "Attribute "+name+" value: "+str+" cannot convert to int";

  return val;
}

double node_att_as_double(const xmlNodePtr p, const string &name)
{
  double val = 0;
  string str = node_att_as_string(p, name);
  if (!string_to_double(str, val))
    throw "Attribute "+name+" value: "+str+" cannot convert to double";

  return val;
}

} // end namespace XMLUtil

} // End namespace SCIRun
