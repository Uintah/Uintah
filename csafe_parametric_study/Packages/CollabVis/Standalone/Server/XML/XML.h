/*
 *
 * XML: Global info for XML parsing and creation
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __XML_H_
#define __XML_H_

#include <map>

#ifdef __sgi
#pragma set woff 1375
#pragma set woff 3303
#pragma set woff 3201
#pragma set woff 1424
#endif

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/deprecated/DOMString.hpp>

#ifdef __sgi
#pragma reset woff 1375
#pragma reset woff 3303
#pragma reset woff 3201
#pragma reset woff 1424
#endif

#include <iostream>

#include <Thread/Mutex.h>

namespace SemotusVisum {
namespace XML {
using namespace std;
using namespace SCIRun;
typedef DOMString  String;

/**************************************
 
CLASS
   XMLI
   
KEYWORDS
   XML
   
DESCRIPTION

   XMLI provides initialization routines for the XML subsystem.
   
****************************************/

class XMLI {
public:
  
  //////////
  // Returns the char* version of the data in String s.
  inline static char * getChar( const String& s ) {
    char * out = NULL;
    mutex.lock();
    out = strdup( s.transcode() );
    mutex.unlock();
    return out;
  }
  
  //////////
  // Initialization routine for XML. Must be called before creating or
  // parsing a document.
  static void initialize() {
    bool initialized = false;
    
    if ( initialized ) return;
    
    try 
    {
      XMLPlatformUtils::Initialize();
    }
    
    catch(const XMLException& toCatch)
    {
      char *pMsg = XMLString::transcode(toCatch.getMessage());
      cerr << "Error during Xerces-c Initialization.\n"
	   << "  Exception message:"
	   << pMsg;
      delete pMsg;
    }
    initialized = true;
  }
  
protected:
  //////////
  // Constructor
  XMLI() {}

  //////////
  // Destructor
  ~XMLI() {}

  static Mutex mutex;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:55  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:54  simpson
// Adding CollabVis files/dirs
//
// Revision 1.8  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.7  2001/08/24 19:14:33  luke
// Fixed XML iostream problem
//
// Revision 1.6  2001/07/17 23:22:37  luke
// Sample server more stable. Now we can send geom or image data to multiple clients.
//
// Revision 1.5  2001/04/04 21:35:33  luke
// Added XML initialization to reader and writer constructors
//
// Revision 1.4  2001/02/08 23:53:33  luke
// Added network stuff, incorporated SemotusVisum namespace
//
// Revision 1.3  2001/01/31 20:45:34  luke
// Changed Properties to Attributes to avoid name conflicts with client and server properties
//
// Revision 1.2  2001/01/29 18:48:47  luke
// Commented XML
//
