#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class IntVector;
  class Vector;
  class Point;
}

class DOMNode;
class DOMText;
class DOMElement;
namespace Uintah {

class TypeDescription;

using std::string;
using std::vector;
using std::map;
using std::ostream;

using SCIRun::IntVector;
using SCIRun::Vector;
using SCIRun::Point;

// This is the "base" problem spec.  There should be ways of breaking
// this up

/**************************************

CLASS
   ProblemSpec
   
   Short description...

GENERAL INFORMATION

   ProblemSpec.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Problem_Specification

DESCRIPTION
   The ProblemSpec class is used as a wrapper of the Xerces DOM implementation.
   
  
WARNING
  
****************************************/

   class ProblemSpec : public RefCounted {
   public:
     
      enum NodeType {
	ELEMENT_NODE = 1, ATTRIBUTE_NODE, TEXT_NODE, CDATA_SECTION_NODE,
	ENTITY_REFERENCE_NODE, ENTITY_NODE, PROCESSING_INSTRUCTION_NODE,
	COMMENT_NODE, DOCUMENT_NODE, DOCUMENT_TYPE_NODE, 
	DOCUMENT_FRAGMENT_NODE, NOTATION_NODE};
     
      inline ProblemSpec(const DOMNode* node, bool doWrite=true){
	d_node = const_cast<DOMNode*>(node); d_write = doWrite; 
      }

      virtual ~ProblemSpec();

      /****************
	 Methods to find a particular Node
      *****************/

      //////////
      // Creates a new document and returns the document element
      static ProblemSpecP createDocument(const std::string& name);

      //////////
      // find the first child node with given node name
      ProblemSpecP findBlock(const std::string& name) const;

      //////////
      // finds the first non-text child node
      ProblemSpecP findBlock() const;

      //////////
      // finds the next sibling node with given node name
      ProblemSpecP findNextBlock(const std::string& name) const;

      //////////
      // find next non-text sibling node
      ProblemSpecP findNextBlock() const;

      //////////
      // find fist child text node
      ProblemSpecP findTextBlock();

      //////////
      // return first child node, null if none
      ProblemSpecP getFirstChild();

      //////////
      // returns the next sibling node, null if none
      ProblemSpecP getNextSibling();
   

      //////////
      // adds a newline with 'tabs' tabs, then creates, appends, and returns
      // a node with name str
      ProblemSpecP appendChild(const char *str, int tabs = 0);

      //////////
      // appends only a node to the tree
      void appendChild(ProblemSpecP pspec);

      //////////
      // Replaces the child node replaced with the node toreplace
      ProblemSpecP replaceChild(ProblemSpecP toreplace, ProblemSpecP replaced);

      //////////
      // Removes the specified child node from the tree
      ProblemSpecP removeChild(ProblemSpecP child);

      //////////
      // Creates and returns a node with name str
      ProblemSpecP createElement(char* str);

      //////////
      // creates (and does not append) a copy of node src
      // deep specifies whether or not to recursively copy the children nodes
      ProblemSpecP importNode(ProblemSpecP src, bool deep);


      /*************
	Methods involving node's attributes
      *************/

      //////////
      // passes back a map of the attributes of this node into value
      void getAttributes(std::map<std::string,std::string>& value);

      //////////
      // passes back the string associated with value of this node's
      // attributes into result
      bool getAttribute(const std::string& value, std::string& result);

      //////////
      // adds an attribute of specified name and value to this node's 
      // attribute list
      void setAttribute(const std::string& name, const std::string& value);

      /*************
	Methods involving node information
      *************/
      

      //////////
      // return the name of this node
      string getNodeName() const;

      //////////
      // return the value of this node (the part between the xml tags)
      string getNodeValue();

      //////////
      // return type of node, as specified in the NodeType enum above
      short getNodeType();



      //////////
      // releases all memory resources associated with this tree
      // only to be used when finished using entire document
      void releaseDocument();


      //////////
      // simply adds the text to this node
      void appendText(const char* str);


      //////////
      // append a node with given value (of varied types) to this node
      void appendElement(const char* name, const std::string& value);
      void appendElement(const char* name, const char* value);
      void appendElement(const char* name, int value);
      void appendElement(const char* name, long value);
      void appendElement(const char* name, const IntVector& value);
      void appendElement(const char* name, const Point& value);
      void appendElement(const char* name, const Vector& value);
      void appendElement(const char* name, double value);
      void appendElement(const char* name, const vector<double>& value);
      void appendElement(const char* name, const vector<int>& value);
      void appendElement(const char* name, bool value);



      //////////
      // Finds child node with name 'name' and passes its value back into
      // value.  If there is no node, an exception is thrown
      void require(const std::string& name, double& value);
      void require(const std::string& name, int& value);
      void require(const std::string& name, long& value);
      void require(const std::string& name, bool& value);
      void require(const std::string& name, std::string& value);
      void require(const std::string& name, IntVector& value);
      void require(const std::string& name, Vector& value);
      void require(const std::string& name, Point& value);
      void require(const std::string& name, vector<double>& value);
      void require(const std::string& name, vector<int>& value); 

      //////////
      // Get any optional attributes associated with a tag
      void requireOptional(const std::string& name, std::string& value);
      ProblemSpecP getOptional(const std::string& name, std::string& value);

      //////////
      // look for the value of tag named name and passes
      // it back into value.  Returns 'this' if found, otherwise null
      ProblemSpecP get(const std::string& name, double& value);
      ProblemSpecP get(const std::string& name, int& value);
      ProblemSpecP get(const std::string& name, long& value);
      ProblemSpecP get(const std::string& name, bool& value);
      ProblemSpecP get(const std::string& name, std::string& value);
      ProblemSpecP get(const std::string& name, IntVector& value);
      ProblemSpecP get(const std::string& name, Vector& value);
      ProblemSpecP get(const std::string& name, Point& value);
      ProblemSpecP get(const std::string& name, vector<double>& value);   
      ProblemSpecP get(const std::string& name, vector<int>& value); 

      //////////
      // look for the first child text node and passes it back to
      // value
      bool get(int &value);
      bool get(long &value);

      //////////
      // look for the value of tag named name and passes
      // it back into value.  If the value isn't there it will create a 
      // node and insert it based on the default value
      ProblemSpecP getWithDefault(const std::string& name, double& value, double defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, int& value, int defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, bool& value, bool defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, std::string& value, const std::string& defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, IntVector& value, const IntVector& defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, Vector& value, const Vector& defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, Point& value, const Point& defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, vector<double>& value, const vector<double>& defaultVal);   
      ProblemSpecP getWithDefault(const std::string& name, vector<int>& value, const vector<int>& defaultVal); 

      //////////
      // Add a stylesheet to this document, to be output at the top of the page
      void addStylesheet(char* type, char* value);

      //////////
      // Output the DOMTree.  
      void output(char* filename = 0) const;

      inline bool operator == (const ProblemSpec& a) const {
	return a.d_node == d_node;
      }
      inline bool operator != (const ProblemSpec& a) const {
	return a.d_node != d_node;
      }
      inline bool operator == (int a) const {
	ASSERT(a == 0);
        a=a;     // This quiets the MIPS compilers
	return d_node == 0;
      }
      inline bool operator != (int a) const {
	ASSERT(a == 0);
        a=a;     // This quiets the MIPS compilers
	return d_node != 0;
      }
      static const TypeDescription* getTypeDescription();
      DOMNode* getNode() const {
	return d_node;
      }

      bool isNull() {
	return (d_node == 0);
      }

      //////////
      // setter and getter of d_write variable
      void writeMessages(bool doWrite) {
        d_write = doWrite;
      }
      bool doWriteMessages() const
      { return d_write; }
   private:

      //////////
      // copy constructor and assignment operator
      ProblemSpec(const ProblemSpec&);
      ProblemSpec& operator=(const ProblemSpec&);
      
      /////////
      // the node
      DOMNode* d_node;
      bool d_write;
   };
   
} // End namespace Uintah

//////////
// to output the document
// Needs to be declared here or causes other code to not compile
std::ostream& operator<<(std::ostream& out, const Uintah::ProblemSpecP pspec);
std::ostream& operator<<(std::ostream& out, const DOMNode* toWrite);
std::ostream& operator<<(std::ostream& out, const DOMText* toWrite);

#endif
