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


#ifndef NEXUS_SP_CHANNEL_H
#define NEXUS_SP_CHANNEL_H

#include <Core/CCA/Comm/SpChannel.h>
#include <stdio.h>
#undef IOV_MAX
#include <globus_nexus.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  /**************************************
 
  CLASS
     NexusSpChannel
   
  DESCRIPTION
     A Globus-Nexus implementation of the SpChannel
     abstract interface. 

  SEE ALSO
     SpChannel.h

  ****************************************/

  class NexusSpMessage;
  class NexusEpMessage;

  class NexusSpChannel : public SpChannel {
  public:

    NexusSpChannel();
    virtual ~NexusSpChannel();
    NexusSpChannel(NexusSpChannel& );
    void openConnection(const URL& );
    void closeConnection();
    Message* getMessage();  
    SpChannel* SPFactory(bool deep); 

 
  private:

    //////////////////
    // NexusEpChannel needs this in order to perform 
    // NexusEpChannel::bind() properly
    friend class NexusEpChannel;

    ////////////////////
    // The Nexus Message classes need access to the 
    // startpoint in order to marshal it, if that is
    // needed. 
    friend class NexusSpMessage;
    friend class NexusEpMessage;

    //////////
    // The startpoint
    globus_nexus_startpoint_t d_sp;

    void printDebug( const std::string& ); 

    //Toggles on/off whether debugging info gets printed
    static const int kDEBUG=0;

  };
}

#endif

