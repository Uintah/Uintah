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

#ifndef Dataflow_Dataflow_PackageDBHandler_h
#define Dataflow_Dataflow_PackageDBHandler_h 1

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <sax/ErrorHandler.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

namespace SCIRun {

class PackageDBHandler : public ErrorHandler
{
public:
  bool foundError;
  
  PackageDBHandler();
  ~PackageDBHandler();
  
  void warning(const SAXParseException& e);
  void error(const SAXParseException& e);
  void fatalError(const SAXParseException& e);
  void resetErrors();
  
private :
  PackageDBHandler(const PackageDBHandler&);
  void operator=(const PackageDBHandler&);
};

} // End namespace SCIRun

#endif

