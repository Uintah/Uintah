#ifndef UINTAH_HOMEBREW_SimpleErrorHandler_H
#define UINTAH_HOMEBREW_SimpleErrorHandler_H

#ifdef __sgi
#define IRIX
#endif
#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif
#include <sax/ErrorHandler.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>

namespace PSECore {
   namespace XMLUtil {
   
   /**************************************
     
     CLASS
       SimpleErrorHandler
      
       Short Description...
      
     GENERAL INFORMATION
      
       SimpleErrorHandler.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       SimpleErrorHandler
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
class SimpleErrorHandler : public ErrorHandler {
public:
   bool foundError;

   SimpleErrorHandler();
   ~SimpleErrorHandler();

   void warning(const SAXParseException& e);
   void error(const SAXParseException& e);
   void fatalError(const SAXParseException& e);
   void resetErrors();

   private :
   SimpleErrorHandler(const SimpleErrorHandler&);
   void operator=(const SimpleErrorHandler&);
};
   } // end namespace XMLUtil
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/20 08:04:28  sparker
// Added XML helper library
//
//

#endif

