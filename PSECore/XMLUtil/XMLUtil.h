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
   } // end namespace XMLUtil
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/06/15 19:51:58  sparker
// Added appendElement for Vector
//
// Revision 1.1  2000/05/20 08:04:28  sparker
// Added XML helper library
//
//

#endif
