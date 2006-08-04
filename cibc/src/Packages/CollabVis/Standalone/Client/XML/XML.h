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
#include <Util/stringUtil.h>

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

using namespace std;
using namespace SCIRun;

/// Makes it easier to write string
typedef DOMString  String;


/**
 * XMLI provides initialization routines for the XML subsystem.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class XMLI {
public:
  
  /**
   * Returns the string version of the data in String s.
   *
   * @param s     DOMString to transcode
   * @return      string version of the data in the input
   */
  inline static string getChar( const String& s ) {
    string out;
    mutex.lock();
    out = s.transcode();
    mutex.unlock();
    return out;
  }
  
  /**
   * Initialization routine for XML. Must be called before creating or
   * parsing a document.
   *
   */
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

  /**
   * Constructor
   *
   */
  XMLI() {}

  /**
   * Destructor
   *
   */
  ~XMLI() {}

  /**
   * Mutex, so multiple threads do not munch each other (as the XML
   * library is not thread-safe */
  static Mutex mutex;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:59  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:05:32  simpson
// Adding CollabVis files/dirs
//
