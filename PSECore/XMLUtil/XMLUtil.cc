
#include <PSECore/XMLUtil/XMLUtil.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <dom/DOM_NamedNodeMap.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;
using namespace SCICore::Geometry;

namespace PSECore {
   namespace XMLUtil {

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
//
//  ostream << DOM_Node   
//
//                Stream out a DOM node, and, recursively, all of its children.
//                This function is the heart of writing a DOM tree out as
//                XML source.  Give it a document node and it will do the whole thing.
//
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
	 
	 //
	 //  Test for the presence of children, which includes both
	 //  text content and nested elements.
	 //
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
	    //
	    //  There were no children.  Output the short form close of the
	    //  element start tag, making it an empty-element tag.
	    //
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
//
//  outputContent  - Write document content from a DOMString to a C++ ostream.
//                   Escape the XML special characters (<, &, etc.) unless this
//                   is suppressed by the command line option.
//
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
//
//  ostream << DOMString    Stream out a DOM string.
//                          Doing this requires that we first transcode
//                          to char * form in the default code page
//                          for the system
//
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
      
}
}
