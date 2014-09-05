/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
    //NOT IMPLEMENTED YET
    //static Object::pointer objectFrom(const Reference&, int mysize = 1, int myrank = 0);

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

    //test if a thread is in the same address space as the master framework
    static bool isFramework();

    //////////
    // Rank and size for parallel proxies are statically set 
    // into these variables. Set by PIDL::initialize().
    static int rank;
    static int size;
    static bool isfrwk;

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




