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
 *  CommNexus.h: Include a bunch of CommNexus files for external clients
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_CommNexus_CommNexus_h
#define CCA_CommNexus_CommNexus_h

#include <globus_nexus.h>

namespace SCIRun {

/**************************************
 
CLASS
   CommNexus
   
KEYWORDS
   CommNexus
   
DESCRIPTION
   A class to encapsulate several static methods for Nexus.
****************************************/
  class CommNexus {
  public:
    //////////
    // Initialize Nexus
    static void initialize();

    //////////
    // Finalize Nexus
    static void finalize();
  };
} // End namespace SCIRun

#endif




