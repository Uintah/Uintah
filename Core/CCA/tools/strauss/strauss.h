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


/*
 *  strauss.h
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   School of Computing
 *   University of Utah
 *   October, 2003
 *
 *  Copyright (C) 2003 SCI 
 */

#ifndef strauss_h
#define strauss_h

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

#include <SCIRun/SCIRunErrorHandler.h>
#include <fstream>
#include <vector>

#include <Core/CCA/tools/strauss/c++ruby/rubyeval.h>
using namespace std;
class SState;

namespace SCIRun {
  class Strauss {
  public:
    Strauss(string& plugin, string& portspec, 
	    string& header, string& impl);
    ~Strauss();
    void emitHeader();
    void emitImpl();
  private:
    Strauss();    

    ////////
    // Various private, read from XML and/or emit methods
    void emitMethodSignatures(SState& hdr, DOMDocument* doc, string tagname);
    void emitMethod(SState& out, DOMDocument* doc, string tagname, string objname);
    void emitPortMethod(SState& out, DOMDocument* doc, string tagname);
    void emitProvidesPortSpec(SState& hdr, DOMDocument* pSpec);
    int readXMLAttr(DOMDocument* doc, string tag, string attr);
    int readXML(DOMDocument* doc, string tag);
    int readXMLChildNode(DOMDocument* doc, string parent, string child);
    int rec_readXMLChildNode(DOMNode* cursor, string child);

    ///////
    // Checks if this is an executable script, if it is it executes the script
    // and emits its result.
    void isScriptScratch(std::string line); 

    ///////
    // Name of bridge component class
    string bridgeComponent;

    ///////
    // Xerces XML stuff
    XercesDOMParser pluginParser;
    XercesDOMParser portParser;
    SCIRunErrorHandler handler;

    ///////
    // Filenames of output files
    string header;
    string impl;
    string plugin;
    string portSpec;

    //////////
    // Collection of file streams that we emit bridge into.
    ofstream fHeader;
    ofstream fImpl;

    //////
    // Scratch pad
    vector<string> scratch;
 
    //////
    // Ruby expression evaluating class   
    RubyEval* ruby; 

  };
} //End of SCIRun namespace

#endif




