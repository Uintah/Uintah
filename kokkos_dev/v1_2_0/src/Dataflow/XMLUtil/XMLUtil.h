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

#ifndef Dataflow_XMLUtil_XMLUtil_H
#define Dataflow_XMLUtil_XMLUtil_H

namespace SCIRun {
  class Point;
  class Vector;
  class IntVector;
}

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <util/XMLUni.hpp>
#ifdef XERCESDEFS_HPP
#include <util/XMLUniDefs.hpp>
#endif
#ifdef __sgi
#pragma reset woff 1375
#endif
#include <string>
#include <iosfwd>

/* NOT_SET is used to indicate active 
   fields inside of data structures that
   represent XML element trees */

#define NOT_SET ((char*)_NOTSET_)

using std::string;


namespace SCIRun {
     
      DOM_Node findNode(const std::string &name,DOM_Node node);
      DOM_Node findNextNode(const std::string& name, DOM_Node node);
      DOM_Node findTextNode(DOM_Node node);
      std::string toString(const XMLCh* const str);
      std::string toString(const DOMString& str);
      void outputContent(std::ostream& target, const DOMString &s);
      std::ostream& operator<<(std::ostream& target, const DOMString& toWrite);
      std::ostream& operator<<(std::ostream& target, const DOM_Node& toWrite);
      void appendElement(DOM_Element& root, const DOMString& name,
			 const std::string& value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 int value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 const IntVector& value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 const Point& value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 const Vector& value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 long value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 double value);
      bool get(const DOM_Node& node, int &value);
      bool get(const DOM_Node& node,
	       const std::string& name, int &value);
      bool get(const DOM_Node& node, long &value);
      bool get(const DOM_Node& node,
	       const std::string& name, long &value);
      bool get(const DOM_Node& node,
	       const std::string& name, double &value);
      bool get(const DOM_Node& node,
	       const std::string& name, std::string &value);
      bool get(const DOM_Node& node,
	       const std::string& name, Vector& value);
      bool get(const DOM_Node& node,
	       const std::string& name, Point& value);
      bool get(const DOM_Node& node,
	       const std::string& name, IntVector &value);
      bool get(const DOM_Node& node,
	       const std::string& name, bool &value);

      extern const char _NOTSET_[];      
      
      //////////////////////////////
      // getSerializedAttributes()
      // returns a string that has an XML format
      // which represents the attributes of "node" 
      
      char* getSerializedAttributes(DOM_Node& node);
      
      
      //////////////////////////////
      // getSerializedChildren()
      // returns a string in XML format that
      // represents the children of the node
      // named "node".
      
      char* getSerializedChildren(DOM_Node& node);
      
      
      string xmlto_string(const DOMString& str);
      string xmlto_string(const XMLCh* const str);
      void invalidNode(const DOM_Node& n, const string& filename);
      DOMString findText(DOM_Node& node);
	

} // End namespace SCIRun

#endif
