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

namespace SCIRun {
   
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
} // End namespace SCIRun


#endif

