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
#include <dom/DOM_NamedNodeMap.hpp>
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

DOM_Node findNode(const std::string &name,DOM_Node node)
{
  // Convert string name to a DOMString;
  
  DOMString search_name(name.c_str());
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
    //DOM_Node tmp = findNode(name,child);
    child = child.getNextSibling();
  }
  
  DOM_Node unknown;
  return unknown;
}

DOM_Node findNextNode(const std::string& name, DOM_Node node)
{
  // Iterate through all of the child nodes that have this name
  DOM_Node found_node = node.getNextSibling();

  DOMString search_name(name.c_str());
  while(found_node != 0){
    DOMString node_name = found_node.getNodeName();
    if (search_name.equals(node_name) ) {
      break;
    }
    found_node = found_node.getNextSibling();
  }
  return found_node;
}


DOM_Node findTextNode(DOM_Node node)
{
   for (DOM_Node child = node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 return child;
      }
   }
  DOM_Node unknown;
  return unknown;   
}

string toString(const XMLCh* const str)
{
    char* s = XMLString::transcode(str);
    if(!s)
       return "";
    string ret = string(s);
    delete[] s;
    return ret;
}

string toString(const DOMString& str)
{
    char* s = str.transcode();
    if(!s)
       return "";
    string ret = string(s);
    delete[] s;
    return ret;
}

// ---------------------------------------------------------------------------
//  ostream << DOM_Node   
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOM_Node& toWrite)
{
   // Get the name and value out for convenience
   DOMString   nodeName = toWrite.getNodeName();
   DOMString   nodeValue = toWrite.getNodeValue();
   
   switch (toWrite.getNodeType()) {
   case DOM_Node::TEXT_NODE:
      {
	 outputContent(target, nodeValue);
	 break;
      }
   
   case DOM_Node::PROCESSING_INSTRUCTION_NODE :
      {
	 target  << "<?"
		 << nodeName
		 << ' '
		 << nodeValue
		 << "?>";
	 break;
      }
   
   case DOM_Node::DOCUMENT_NODE :
      {
	 // Bug here:  we need to find a way to get the encoding name
	 //   for the default code page on the system where the
	 //   program is running, and plug that in for the encoding
	 //   name.  
	 target << "<?xml version='1.0' encoding='ISO-8859-1' ?>\n";
	 DOM_Node child = toWrite.getFirstChild();
	 while( child != 0)
            {
	       target << child << endl;
	       child = child.getNextSibling();
            }
	 
	 break;
      }
   
   case DOM_Node::ELEMENT_NODE :
      {
	 // Output the element start tag.
	 target << '<' << nodeName;
	 
	 // Output any attributes on this element
	 DOM_NamedNodeMap attributes = toWrite.getAttributes();
	 int attrCount = attributes.getLength();
	 for (int i = 0; i < attrCount; i++) {
	    DOM_Node  attribute = attributes.item(i);
	    
	    target  << ' ' << attribute.getNodeName()
		    << " = \"";
	    //  Note that "<" must be escaped in attribute values.
	    outputContent(target, attribute.getNodeValue());
	    target << '"';
	 }
	 
	 //  Test for the presence of children, which includes both
	 //  text content and nested elements.
	 DOM_Node child = toWrite.getFirstChild();
	 if (child != 0) {
	    // There are children. Close start-tag, and output children.
	    target << ">";
	    while( child != 0) {
	       target << child;
	       child = child.getNextSibling();
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
   
   case DOM_Node::ENTITY_REFERENCE_NODE:
      {
	 DOM_Node child;
	 for (child = toWrite.getFirstChild(); child != 0; child = child.getNextSibling())
	    target << child;
	 break;
      }
   
   case DOM_Node::CDATA_SECTION_NODE:
      {
	 target << "<![CDATA[" << nodeValue << "]]>";
	 break;
      }
   
   case DOM_Node::COMMENT_NODE:
      {
	 target << "<!--" << nodeValue << "-->";
	 break;
      }
   
   default:
      cerr << "Unrecognized node type = "
	   << (long)toWrite.getNodeType() << endl;
   }
   return target;
}


// ---------------------------------------------------------------------------
//  outputContent  - Write document content from a DOMString to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
// ---------------------------------------------------------------------------
void outputContent(ostream& target, const DOMString &toWrite)
{
   int            length = toWrite.length();
   const XMLCh*   chars  = toWrite.rawBuffer();
   
   int index;
   for (index = 0; index < length; index++) {
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
	 target << toWrite.substringData(index, 1);
	 break;
      }
   }
}


// ---------------------------------------------------------------------------
//  ostream << DOMString    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
// ---------------------------------------------------------------------------
ostream& operator<<(ostream& target, const DOMString& s)
{
   char *p = s.transcode();
   target << p;
   delete [] p;
   return target;
}

void appendElement(DOM_Element& root, const DOMString& name,
		   const std::string& value)
{
   DOM_Text leader = root.getOwnerDocument().createTextNode("\n\t");
   root.appendChild(leader);
   DOM_Element newElem = root.getOwnerDocument().createElement(name);
   root.appendChild(newElem);
   DOM_Text newVal = root.getOwnerDocument().createTextNode(value.c_str());
   newElem.appendChild(newVal);
   DOM_Text trailer = root.getOwnerDocument().createTextNode("\n");
   root.appendChild(trailer);
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   int value)
{
   ostringstream val;
   val << value;
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   const IntVector& value)
{
   ostringstream val;
   val << '[' << value.x() << ", " << value.y() << ", " << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   const Point& value)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   const Vector& value)
{
   ostringstream val;
   val << '[' << setprecision(17) << value.x() << ", " << setprecision(17) << value.y() << ", " << setprecision(17) << value.z() << ']';
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   long value)
{
   ostringstream val;
   val << value;
   appendElement(root, name, val.str());
}
      
void appendElement(DOM_Element& root, const DOMString& name,
		   double value)
{
   ostringstream val;
   val << setprecision(17) << value;
   appendElement(root, name, val.str());
}
      
bool get(const DOM_Node& node, int &value)
{
   for (DOM_Node child = node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char* s = val.transcode();
	 value = atoi(s);
	 delete[] s;
	 return true;
      }
   }
   return false;
}

bool get(const DOM_Node& node,
		const std::string& name, int &value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
   return get(found_node, value);
}

bool get(const DOM_Node& node, long &value)
{
   for (DOM_Node child = node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char* s = val.transcode();
	 value = atoi(s);
	 delete[] s;
	 return true;
      }
   }
   return false;
}

bool get(const DOM_Node& node,
		const std::string& name, long &value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
   return get(found_node, value);
}

bool get(const DOM_Node& node,
		const std::string& name, double &value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
   for (DOM_Node child = found_node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char* s = val.transcode();
	 value = atof(s);
	 delete[] s;
	 return true;
      }
   }
   return false;
}

bool get(const DOM_Node& node,
		const std::string& name, std::string &value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
   for (DOM_Node child = found_node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char *s = val.transcode();
	 value = std::string(s);
	 delete[] s;
	 return true;
      }
   }
   return false;
}

bool get(const DOM_Node& node,
		const std::string& name, 
		Vector& value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
   for (DOM_Node child = found_node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char *s = val.transcode();
	 string string_value = std::string(s);
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
	 return true;
      }
   }
   return false;
}

bool get(const DOM_Node& node,
		const std::string& name, 
		Point& value)
{
   Vector v;
   bool status=get(node, name, v);
   value=Point(v);
   return status;
}

bool get(const DOM_Node& node,
		const std::string& name, 
		IntVector &value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
   for (DOM_Node child = found_node.getFirstChild(); child != 0;
	child = child.getNextSibling()) {
      if (child.getNodeType() == DOM_Node::TEXT_NODE) {
	 DOMString val = child.getNodeValue();
	 char *s = val.transcode();
	 string string_value = std::string(s);
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
	 return true;
      }
   }
   return false;
}

bool get(const DOM_Node& node,
		const std::string& name, bool &value)
{
   DOM_Node found_node = findNode(name, node);
   if(found_node.isNull())
      return false;
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
	
	 return true;
      }
   }
   return false;
}

char* getSerializedAttributes(DOM_Node& d)
{
  char* string = 0;
  char* fullstring = new char[1];
  char* newstring = 0;

  fullstring[0]='\0';

  DOM_NamedNodeMap attr = d.getAttributes();
  int length = attr.getLength();
  int index = 0;
  for(DOM_Node n=attr.item(index);
      index!=length;n=attr.item(++index)) {
    string = new char[strlen(n.getNodeName().transcode())+
		      strlen(n.getNodeValue().transcode())+5];
    sprintf(string," %s=\"%s\"",n.getNodeName().transcode(),
	    n.getNodeValue().transcode());

    int newlength = strlen(string)+strlen(fullstring);
    newstring = new char[newlength+1];
    newstring[0]='\0';
    sprintf(newstring,"%s%s",fullstring,string);
    delete[] fullstring;
    fullstring = newstring;
    newstring = 0;

    delete[] string;
    string = 0;
  }

  return fullstring;
}

char* getSerializedChildren(DOM_Node& d)
{
  char* temp = 0;
  char* temp2 = 0;
  char* string = 0;
  char* fullstring = new char[1];
  char* newstring = 0;

  fullstring[0]='\0';

  for (DOM_Node n=d.getFirstChild();n!=0;n=n.getNextSibling()) {
    if (n.getNodeType()==DOM_Node::TEXT_NODE) {
      string = new char[strlen(n.getNodeValue().transcode())+1];
      string[0]='\0';
      sprintf(string,"%s",n.getNodeValue().transcode());
    } else {
      temp = getSerializedAttributes(n);
      temp2 = getSerializedChildren(n);
      string = new char[2*strlen(n.getNodeName().transcode())+
		        strlen(temp)+strlen(temp2)+6];
      string[0]='\0';
      sprintf(string,"<%s%s>%s</%s>",n.getNodeName().transcode(),
	      temp,temp2,n.getNodeName().transcode());
      delete[] temp;
      delete[] temp2;
    } 
    
    int newlength = strlen(string) + strlen(fullstring);
    newstring = new char[newlength+1];
    newstring[0]='\0';
    
    sprintf(newstring,"%s%s",fullstring,string);
    delete[] fullstring;
    fullstring = newstring;
    newstring = 0;
    
    delete[] string;
    string = 0;
  }
  
  return fullstring;
}

string xmlto_string(const DOMString& str)
{
  char* s = str.transcode();
  string ret = string(s);
  delete[] s;
  return ret;
}

string xmlto_string(const XMLCh* const str)
{
  char* s = XMLString::transcode(str);
  string ret = string(s);
  delete[] s;
  return ret;
}

void invalidNode(const DOM_Node& n, const string& filename)
{
  if(n.getNodeType() == DOM_Node::COMMENT_NODE)
      return;
  if(n.getNodeType() == DOM_Node::TEXT_NODE){
    DOMString s = n.getNodeValue();
    char* str = s.transcode();
    bool allwhite=true;
    for(char* p = str; *p != 0; p++){
      if(!isspace(*p))
	allwhite=false;
      }
    if(!allwhite){
      postMessage(string("Extraneous text: ")+str+"after node: "+xmlto_string(n.getNodeName())+"(in file "+filename+")");
    }
    delete[] str;
    return;
  }
  postMessage(string("Do not understand node: ")+xmlto_string(n.getNodeName())+"(in file "+filename+")");
}

DOMString findText(DOM_Node& node)
{
  for(DOM_Node n = node.getFirstChild();n != 0; n = n.getNextSibling()){
    if(n.getNodeType() == DOM_Node::TEXT_NODE)
      return n.getNodeValue();
  }
  return 0;
}

} // End namespace SCIRun
