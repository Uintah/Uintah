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

#ifndef CCA_PIDL_PIDL_h
#define CCA_PIDL_PIDL_h

#include <Core/CCA/PIDL/Object.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

  class SpChannel;
  class URL;
  class IntraComm;
/**************************************
 
CLASS
   PIDL
   
KEYWORDS
   PIDL
   
DESCRIPTION
   A class to encapsulate several static methods for PIDL.
****************************************/

  class DataTransmitter;
  class PIDL {
  public:
    //////////
    // Initialize PIDL
    static void initialize(int rank=0, int size=1);

    //////////
    // Finalize PIDL
    static void finalize();

    //////////
    // Get the start point channel  
    static SpChannel* getSpChannel();

    //////////
    // Get the start point channel
    static EpChannel* getEpChannel();

    //////////
    // Create a base Object class from the given URL
    static Object::pointer objectFrom(const URL&);

    //////////
    // Create a base Object class from the given URL array
    static Object::pointer objectFrom(const int urlc, const URL urlv[], int mysize = 1, int myrank = 0);
      
    //////////
    // Create a base Object class from the given URL vector
    static Object::pointer objectFrom(const std::vector<URL>& urlv, int mysize = 1, int myrank = 0);

    //////////
    // Create a base Object class from the given Reference
    static Object::pointer objectFrom(const Reference&);

    //////////
    // Create a base Object class from a vector of proxies  
    static Object::pointer objectFrom(const std::vector<Object::pointer>& pxy, int mysize = 1, int myrank = 0);

    //////////
    // Go into the main loop which services requests for all
    // objects.  This will not return until all objects have
    // been destroyed.
    static void serveObjects();

    //////////
    // Return the object Warehouse.  Most clients will not
    // need to use this.
    static Warehouse* getWarehouse();

    //////////
    // Get the inter-component communication object
    // (only in parallel components)
    static IntraComm* getIntraComm();

    static DataTransmitter *getDT();

    static bool isNexus();

    //////////
    // Rank and size for parallel proxies are statically set 
    // into these variables. Set by PIDL::initialize().
    static int rank;
    static int size;

    //////////
    // Used to save one proxy in order to call get exception
    // upon PIDL::finalize()
    static bool sampleProxy;
    static Object::pointer* optr;

  protected:
  private:
    //////////
    // Initialize proper communication library to
    // be used throughout the PIDL
    static void setCommunication(int c);

    //////////
    // Initialize a parallel communication library to
    // be used within a component 
    static void setIntraCommunication(int c);

    //////////
    // The warehouse singleton object
    static Warehouse* warehouse;

    static DataTransmitter* theDataTransmitter;

    //////////
    // Private constructor to prevent creation of a PIDL
    PIDL();

  };
} // End namespace SCIRun



#endif




