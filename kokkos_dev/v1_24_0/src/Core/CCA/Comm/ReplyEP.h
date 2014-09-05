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
 *  ReplyEP.h:  A set of prebuilt endpoints for replies
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_Comm_ReplyEP_h
#define CCA_Comm_ReplyEP_h

#include <Core/Thread/Semaphore.h>
#include <stdio.h>
#undef IOV_MAX
#include <globus_nexus.h>


namespace SCIRun {
  /**************************************
 
  CLASS
     ReplyEP
   
  KEYWORDS
     Endpoint, PIDL
   
  DESCRIPTION
     Class to provide efficient access to a transient endpoint, typically
     used for a reply.  The static methods acquire and release are used
     to obtain a ReplyEP for use in a reply.  The wait() method blocks
     until the reply has been sent to the reply handler (handler number
     0), and returns the nexus buffer.
  ****************************************/
  class ReplyEP {
  public:
    //////////
    // Acquire a ReplyEP.  Creates a new one if none are
    // ready to be used.  Eventually, the caller should call
    // release() on this object.
    static ReplyEP* acquire();

    //////////
    // Release the ReplyEP.
    static void release(ReplyEP*);
    
    //////////
    // Wait for the reply and return the associated buffer.  The
    // caller should eventually call globus_nexus_buffer_destroy
    // on this buffer.
    globus_nexus_buffer_t wait();

    //////////
    // Return the startpoint associated with this object.  This
    // is used only in sidl-generated stub code.  It does NOT
    // call globus_nexus_startpoint_copy.  That should be done
    // in the generated code if necessary.
    void get_startpoint_copy(globus_nexus_startpoint_t*);


    //////////
    // The nexus handler for this object.  This is for internal
    // use only.
    static void reply_handler(globus_nexus_endpoint_t* ep,
			      globus_nexus_buffer_t* buffer,
			      globus_bool_t threadedHandler);
    
    //////////
    // Destructor which needs to be public as sometimes we don't want
    // to put a ReplyEP back in the pool
    ~ReplyEP();

  private:
    //////////
    // Create the ReplyEP object. Used internally
    ReplyEP();

    //////////
    // Copy construct is private to prevent accidental use.
    ReplyEP(const ReplyEP&);

    //////////
    // The endpoint
    globus_nexus_endpoint_t d_ep;

    //////////
    // The startpoint associated with this endpoint
    globus_nexus_startpoint_t d_sp;

    //////////
    // The semaphore used for blocking
    SCIRun::Semaphore d_sema;

    //////////
    // A place to bounce the buffer from the handler thread to
    // the waiter.
    globus_nexus_buffer_t d_bufferHandoff;

  };
}

#endif
