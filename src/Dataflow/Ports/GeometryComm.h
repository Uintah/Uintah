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
 *  GeometryComm.h: Communication classes for Geometry
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_GeometryComm_h
#define SCI_Datatypes_GeometryComm_h 1

#include <Dataflow/share/share.h>

#include <Dataflow/Comm/MessageBase.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/View.h>
#include <Core/Thread/FutureValue.h>

namespace SCIRun {
  class GeomObj;
  class Semaphore;


struct GeomReply {
    int portid;
    int* busy_bit;
    GeomReply();
    GeomReply(int, int*);
};

class PSECORESHARE GeometryComm : public MessageBase {
public:
    GeometryComm(Mailbox<GeomReply>*);
    GeometryComm(int, GeomID, GeomObj*, const string&, CrowdMonitor* lock);
    GeometryComm(int, GeomID, int del);
    GeometryComm(MessageTypes::MessageType, int);
    GeometryComm(MessageTypes::MessageType, int, Semaphore* wait);
    GeometryComm(MessageTypes::MessageType, int, int, View);
    GeometryComm(MessageTypes::MessageType, int portid,
		 FutureValue<GeometryData*>* reply,
		 int which_viewwindow, int datamask);
    GeometryComm(MessageTypes::MessageType, int portid,
		 FutureValue<int>* reply);
    virtual ~GeometryComm();

    Mailbox<GeomReply>* reply;
    int portno;
    GeomID serial;
    GeomObj* obj;
    string name;
    CrowdMonitor* lock;
    Semaphore* wait;
    int del;
    View view;

    GeometryComm* next;

    int which_viewwindow;
    int datamask;
    FutureValue<GeometryData*>* datareply;
    FutureValue<int>* nreply;
};

} // End namespace SCIRun


#endif /* SCI_Datatypes_GeometryComm_h */
