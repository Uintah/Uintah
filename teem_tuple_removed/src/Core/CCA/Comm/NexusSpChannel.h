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

