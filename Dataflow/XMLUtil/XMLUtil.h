#ifndef PSECore_XMLUtil_XMLUtil_H
#define PSECore_XMLUtil_XMLUtil_H

namespace SCICore {
   namespace Geometry {
      class Point;
      class Vector;
      class IntVector;
   }
}

#include <SCICore/Containers/String.h>

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

namespace PSECore {
   namespace XMLUtil {
     using SCICore::Geometry::Point;
     using SCICore::Geometry::Vector;
     using SCICore::Geometry::IntVector;
     using SCICore::Containers::clString;
     
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
			 const SCICore::Geometry::IntVector& value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 const SCICore::Geometry::Point& value);
      void appendElement(DOM_Element& root, const DOMString& name,
			 const SCICore::Geometry::Vector& value);
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
      //
      // getSerializedAttributes()
      //
      // returns a string that has an XML format
      // which represents the attributes of "node" 
      // 
      
      char* getSerializedAttributes(DOM_Node& node);
      
      
      //////////////////////////////
      //
      // getSerializedChildren()
      //
      // returns a string in XML format that
      // represents the children of the node
      // named "node".
      //
      
      char* getSerializedChildren(DOM_Node& node);
      
      
      //////////////////////////////
      //
      // removeWhiteSpace()
      //
      // removes all leading and trailing
      // white space from "string".
      // Returns "string" (after it has
      // been modified).
      //
      
      char* removeWhiteSpace(char* string);

      
      clString xmlto_string(const DOMString& str);
      clString xmlto_string(const XMLCh* const str);
      void invalidNode(const DOM_Node& n, const clString& filename);
      DOMString findText(DOM_Node& node);
	

   } // end namespace XMLUtil
} // end namespace PSECore

#endif
