/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include <Core/Util/Handle.h>
#include <Core/Util/RefCounted.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

#include <string>
#include <vector>
#include <map>

typedef struct _xmlNode xmlNode;

namespace Uintah {

class TypeDescription;

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
  

KEYWORDS
   Problem_Specification

DESCRIPTION
   The ProblemSpec class is used as a wrapper of the libxml xml implementation.
   
  
WARNING
  
****************************************/

class ProblemSpec : public RefCounted {
public:
     
  // 'checkForInputError()' determines if the input string is a valid int or float.
  //  If the string is not valid, a ProblemSetupException is thrown.
  enum CheckType { INT_TYPE, FLOAT_TYPE };
  static void checkForInputError( const std::string & stringValue, CheckType type );
     
  /**
   * \enum InputType
   * \brief Enum that helps in determining the datatype of some input from the input file.
   */
  enum InputType { NUMBER_TYPE,
                   VECTOR_TYPE,
                   STRING_TYPE,
                   UNKNOWN_TYPE
  };

  /**
   *  \brief Function that returns InputType enum with the likely data contained in the string.
   *  \param stringValue The input data that we wish to question for type information.
                         This is typically obtained from ProblemSpec::get("value", stringValue).
   *  \return Returns an InputType enum with the likely data contained in the string. 
   */
  static ProblemSpec::InputType getInputType( const std::string & stringValue );

  enum NodeType {
    ELEMENT_NODE = 1, ATTRIBUTE_NODE, TEXT_NODE, CDATA_SECTION_NODE,
    ENTITY_REFERENCE_NODE, ENTITY_NODE, PROCESSING_INSTRUCTION_NODE,
    COMMENT_NODE, DOCUMENT_NODE, DOCUMENT_TYPE_NODE, 
    DOCUMENT_FRAGMENT_NODE, NOTATION_NODE};

  inline ProblemSpec( const xmlNode* node, bool toplevel = false ) {
    d_node = const_cast<xmlNode*>(node); 
    d_documentNode = toplevel;
  }

  inline virtual ~ProblemSpec() { if (d_documentNode) releaseDocument();}

  /****************
    Methods to find a particular Node
  *****************/

  //////////
  // Creates a new document and returns the document element
  static ProblemSpecP createDocument(const std::string& name);

  //////////
  // find the first child node with given 'name'
  ProblemSpecP findBlock( const std::string & name ) const;

  //////////
  // Find the last child node with given 'name'.  (Usually used to find a node that was just appended.)
  ProblemSpecP findLastBlock( const std::string & name ) const;

  //////////
  // find the first child node with given node name and attribute 
  ProblemSpecP findBlockWithAttribute( const std::string & name,
                                       const std::string & attribute ) const;

  //////////
  // find the first child node with given node name and attribute 
  ProblemSpecP findBlockWithOutAttribute( const std::string & name ) const;

  //////////
  // finds the first non-text child node
  ProblemSpecP findBlock() const;


  //////////
  // finds the next sibling node with given node name
  ProblemSpecP findNextBlock( const std::string & name ) const;

  //////////
  // find next non-text sibling node
  ProblemSpecP findNextBlock() const;

  //////////
  // find first child text node
  ProblemSpecP findTextBlock();

  //////////
  // return first child node, null if none
  ProblemSpecP getFirstChild();

  //////////
  // returns the next sibling node, null if none
  ProblemSpecP getNextSibling();
   
  //////////
  // add a comment
  void addComment(const std::string & comment );

  ProblemSpecP makeComment(const std::string & comment );

  //////////
  // adds a newline with 'tabs' tabs, then creates, appends, and returns
  // a node with name str
  ProblemSpecP appendChild( const char * str );

  //////////
  // appends only a node to the tree
  void appendChild( ProblemSpecP pspec );

  //////////
  // Replaces the child node replaced with the node toreplace
  void replaceChild( ProblemSpecP toreplace, ProblemSpecP replaced );

  //////////
  // Removes the specified child node from the tree
  void removeChild( ProblemSpecP child );

  //////////
  // creates (and does not append) a copy of node src
  // deep specifies whether or not to recursively copy the children nodes
  ProblemSpecP importNode( ProblemSpecP src, bool deep );

  /*************
     Methods involving node's attributes
  *************/

  //////////
  // passes back a map of the attributes of this node into value
  void getAttributes( std::map<std::string,std::string> & value ) const;

  //////////
  // If 'attribute' is found, then 'result' is set to the attribute's value.  If it is not found,
  // then 'result' is not modified.
  bool getAttribute( const std::string & attribute, std::string & result ) const;
  bool getAttribute( const std::string & attribute, std::vector<std::string>& result) const;

  //////////
  // passes back the double value associated with value of this node's
  // attributes into result
  bool getAttribute( const std::string & value, double & result ) const;

  //////////
  // adds an attribute of specified name and value to this node's 
  // attribute list
  void setAttribute( const std::string & name, const std::string & value );

  /*************
     Methods involving node information
  *************/
      
  //////////
  // return the name of this node
  std::string getNodeName() const;

  //////////
  // returns the value of this node (the ascii text between the
  // xml tags) (Not to be confused with the nodes children which
  // would be in <child></child> notation.)
  std::string getNodeValue() const;

  //////////
  // return type of node, as specified in the NodeType enum above
  short getNodeType() const;

  //////////
  // releases all memory resources associated with this tree
  // only to be used when finished using entire document
  void releaseDocument();

  // Returns the root node of the DOM tree.
  ProblemSpecP getRootNode();

  //////////
  // append a node with given value (of varied types) to this node
  void appendElement( const char * name, double              value );
  void appendElement( const char * name, int                 value );
  void appendElement( const char * name, long                value );
  void appendElement( const char * name, bool                value );
  void appendElement( const char * name, const std::string & value );
  void appendElement( const char * name, const char        * value );
  void appendElement( const char * name, const IntVector   & value );
  void appendElement( const char * name, const Point       & value );
  void appendElement( const char * name, const Vector      & value );
  void appendElement( const char * name, const std::vector<double>      & value );
  void appendElement( const char * name, const std::vector<int>         & value );
  void appendElement( const char * name, const std::vector<std::string> & value );

  //////////
  // Finds child node with name 'name' and passes its value back into
  // value.  If there is no node, an exception is thrown
  void require( const std::string & name, double       & value );
  void require( const std::string & name, int          & value );
  void require( const std::string & name, unsigned int & value );
  void require( const std::string & name, long         & value );
  void require( const std::string & name, bool         & value );
  void require( const std::string & name, std::string  & value );
  void require( const std::string & name, IntVector    & value );
  void require( const std::string & name, Vector       & value );
  void require( const std::string & name, Point        & value );
  void require( const std::string & name, std::vector<double>    & value );
  void require( const std::string & name, std::vector<int>       & value) ; 
  void require( const std::string & name, std::vector<IntVector> & value );

  //////////
  // Look for the child tag named 'name' and pass back its
  // 'value'.  Returns true if found.  Has the possiblity of
  // throwing a ProblemSetupException if the data is not the
  // correct type (ie: you call get(name,double) but the tag
  // contains a string).
  //
  bool get( const std::string & name, double       & value ) const;
  bool get( const std::string & name, int          & value ) const;
  bool get( const std::string & name, unsigned int & value ) const;
  bool get( const std::string & name, long         & value ) const;
  bool get( const std::string & name, bool         & value ) const;
  bool get( const std::string & name, std::string  & value ) const;
  bool get( const std::string & name, IntVector    & value ) const;
  bool get( const std::string & name, Vector       & value ) const;
  bool get( const std::string & name, Point        & value ) const;
  bool get( const std::string & name, std::vector<double>      & value, const int nItems = -1 ) const; // parse only nItems separated by comma or space
  bool get( const std::string & name, std::vector<int>         & value ) const;
  bool get( const std::string & name, std::vector<IntVector>   & value ) const;
  bool get( const std::string & name, std::vector<std::string> & value, const int nItems = -1 ) const; // parse only nItems separated by comma or space
      
  void parseIntVector( const std::string & str, IntVector & value ) const;
      
  //////////
  // look for the first child text node and passes it back to
  // value
  bool get( int         & value ) const;
  bool get( long        & value ) const;
  bool get( double      & value ) const;
  bool get( std::string & value ) const;
  bool get( Vector      & value ) const;

  //////////
  // Look for the value of tag 'name' and passes
  // it back into value.  If the value isn't there it will create a 
  // node and insert it based on the default value
  //
  // Types of values that are used with this function include:
  //     double, int, bool, string, IntVector, Vector, Point, vector<double/int/string>

  template<typename Type> bool getWithDefault( const std::string & name, Type & value, const Type & defaultVal );
      void require(const std::string& name, std::vector<std::string>& value);

  //////////
  // Add a stylesheet to this document, to be output at the top of the page
  void addStylesheet( char * type, char * value );

  //////////
  // Output the DOMTree.  
  void output( const char * filename = 0 ) const;

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

  xmlNode* getNode() const { return d_node; }

  bool     isNull() const  { return (d_node == 0); }

  // Returns the name of the file that this problem spec came from... WARNING
  // haven't tested this extensively.  It is in place for 'hacks' that need to
  // deal with data in UDAs that is _NOT_ managed explicitly by the Datawarehouse.
  std::string getFile() const;

  //////////
  // to output the document
  //friend std::ostream& operator<<(std::ostream& out, const Uintah::ProblemSpecP pspec);

  //////////
  // Pretty-prints the DOM tree.  Used for debugging.
  void print() { printHelper( d_node ); }

private:

  //////////
  // copy constructor and assignment operator
  ProblemSpec(const ProblemSpec&);
  ProblemSpec& operator=( const ProblemSpec & );
      
  xmlNode * d_node;
  bool      d_documentNode;

  void printHelper( xmlNode * node, int depth = 0 );
  
};

} // End namespace Uintah

#endif
