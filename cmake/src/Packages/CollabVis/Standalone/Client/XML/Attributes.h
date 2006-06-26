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
#include <Util/stringUtil.h>
#include <iostream>

namespace SemotusVisum {

using namespace std;

/// A map of two strings
typedef map<string,string> cmap;


/**
 * Attributes provides an abstract interface to store and retrieve
 * name/value pairs.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Attributes {
public:

  /**
   * Constructor. Initializes empty map.
   *
   */
  Attributes() {}

  /**
   * Destructor
   *
   */
  ~Attributes() {}

  /**
   * Adds the given key/value pair to the map.
   *
   * @param key       Key of the pair
   * @param value     Value of the pair
   */
  inline void     setAttribute(string key, string value);

  /**
   * Retrieves the value, given the key. Returns "" if the key is not
   * present in the map.
   *
   * @param key       Key to search for
   * @return          Value, or "" if not found.
   */
  inline string   getAttribute(const string key);

  /**
   *  Returns a constant iterator for the map.
   *
   * @return  Constant iterator (read-only)
   */
  inline map<string,string>::const_iterator getIterator() {
    return theMap.begin();
  }

  /**
   * Returns true if the given iterator is at the end of the map.
   *
   * @param it     Constant iterator
   * @return       True if the iterator is at the end of the map.
   */
  inline bool mapEnd(cmap::const_iterator it) const {
    return (it == theMap.end());
  }

  /**
   * Clears the map.
   *
   */
  inline void clear() {
    theMap.clear();
  }

  /**
   *  Returns true if the map is empty; else returns false.
   *
   * @return  True if map is empty; else false.
   */
  inline bool empty() const {
    return theMap.empty();
  }

  /**
   * Dumps the map to stderr.
   *
   */
  inline void list();
protected:
  /// Low-level representation of the map.
  cmap theMap;
  
};


void
Attributes::setAttribute( string key, string value ) {
  theMap.insert( make_pair( key, value ) );
}

string
Attributes::getAttribute( const string key ) {
  for (cmap::const_iterator i = getIterator();
       !mapEnd(i); i++) {
    if ( key == i->first )
      return i->second;
  }
  return "";
  
}

void
Attributes::list() {
  for (cmap::const_iterator i = getIterator();
       !mapEnd(i); i++)
    cerr << i->first << " -> " << i->second << endl; 
}

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
