/*
 *
 * Attributes: Abstraction for < name, value > pairs
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#ifndef __Attributes_H_
#define __Attributes_H_

#include <map>
//#include <iostream>
namespace SemotusVisum {
namespace XML {

using namespace std;

//////////
// A map of two strings
typedef map<char *,char *> cmap;

/**************************************
 
CLASS
   Attributes
   
KEYWORDS
   XML
   
DESCRIPTION

   Attributes provides an abstract interface to store and retrieve
   name/value pairs.
   
****************************************/

class Attributes {
public:

  //////////
  // Constructor. Initializes empty map.
  Attributes() {}

  //////////
  // Destructor
  ~Attributes() {}

  //////////
  // Adds the given key/value pair to the map.
  inline void     setAttribute(char * key, char * value);

  //////////
  // Retrieves the value, given the key. Returns NULL if the key is not
  // present in the map.
  inline char *   getAttribute(char * key);

  //////////
  // Returns a constant iterator for the map.
  inline map<char *,char *>::const_iterator getIterator() {
    return theMap.begin();
  }

  //////////
  // Returns true if the given iterator is at the end of the map.
  inline bool mapEnd(cmap::const_iterator it) const {
    return (it == theMap.end());
  }

  //////////
  // Clears the map.
  inline void clear() {
    theMap.clear();
  }

  //////////
  // Returns true if the map is empty; else returns false.
  inline bool empty() const {
    return theMap.empty();
  }

  inline void list();
protected:
  cmap theMap;
  
};


void
Attributes::setAttribute( char * key, char * value ) {
  theMap.insert( make_pair( key, value ) );
}

char *
Attributes::getAttribute( char * key ) {
  
  for (cmap::const_iterator i = getIterator();
       !mapEnd(i); i++)
    if (!strcmp( key, i->first ))
      return i->second;

  return NULL;
  
}

void
Attributes::list() {
  //std::cerr << "List:" << endl;
  //for (cmap::const_iterator i = getIterator();
  //     !mapEnd(i); i++)
  //  std::cerr << i->first << "\t" << i->second << "-" << endl;
       
  
}

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:55  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:26:53  simpson
// Adding CollabVis files/dirs
//
// Revision 1.8  2001/10/11 16:38:08  luke
// Foo
//
// Revision 1.7  2001/10/10 14:56:23  luke
// Made a few updates so that code would compile/link with SCIRun
//
// Revision 1.6  2001/05/29 03:43:13  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
//
// Revision 1.5  2001/05/12 03:32:49  luke
// Moved driver to new location
//
// Revision 1.4  2001/04/04 21:35:33  luke
// Added XML initialization to reader and writer constructors
//
// Revision 1.3  2001/02/08 23:53:33  luke
// Added network stuff, incorporated SemotusVisum namespace
//
// Revision 1.2  2001/01/31 22:21:56  luke
// Arg...fixed misnaming in Attributes...
//
// Revision 1.1  2001/01/31 20:45:33  luke
// Changed Properties to Attributes to avoid name conflicts with
// client and server properties
//
// Revision 1.2  2001/01/29 18:48:47  luke
// Commented XML
//
