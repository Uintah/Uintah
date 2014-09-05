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
 *  PIDL.h: Include a bunch of PIDL files for external clients
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_PIDL_PIDL_h
#define Component_PIDL_PIDL_h

#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/PIDLException.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/PIDL/pidl_cast.h>
#include <string>

namespace PIDL {

/**************************************
 
CLASS
   PIDL
   
KEYWORDS
   PIDL
   
DESCRIPTION
   A class to encapsulate several static methods for PIDL.
****************************************/
  class PIDL {
  public:
    //////////
    // Initialize PIDL
    static void initialize(int argc, char* argv[]);

    //////////
    // Create a base Object class from the given URL
    static Object objectFrom(const URL&);

    //////////
    // Create a base Object class from the given Reference
    static Object objectFrom(const Reference&);

    //////////
    // Go into the main loop which services requests for all
    // objects.  This will not return until all objects have
    // been destroyed.
    static void serveObjects();

    //////////
    // Return the URL for the current process.  Individual
    // objects may be identified by appending their id number
    // or name to the end of the string.
    static std::string getBaseURL();

    //////////
    // Return the object Warehouse.  Most clients will not
    // need to use this.
    static Warehouse* getWarehouse();
  protected:
  private:
    //////////
    // The warehouse singleton object
    static Warehouse* warehouse;

    //////////
    // Private constructor to prevent creation of a PIDL
    PIDL();
  };
} // End namespace PIDL

#endif

