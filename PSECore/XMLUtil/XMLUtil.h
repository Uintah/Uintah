#ifndef PSECore_XMLUtil_XMLUtil_H
#define PSECore_XMLUtil_XMLUtil_H

namespace SCICore {
   namespace Geometry {
      class Point;
      class Vector;
      class IntVector;
   }
}

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif
#include <string>
#include <iosfwd>

namespace PSECore {
   namespace XMLUtil {
     using SCICore::Geometry::Point;
     using SCICore::Geometry::Vector;
     using SCICore::Geometry::IntVector;
     
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
   } // end namespace XMLUtil
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/06/27 17:08:24  bigler
// Steve moved some functions around for me.
//
// Revision 1.2  2000/06/15 19:51:58  sparker
// Added appendElement for Vector
//
// Revision 1.1  2000/05/20 08:04:28  sparker
// Added XML helper library
//
//

#endif
