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

#include <Dataflow/Comm/MessageBase.h>
#include <Core/Geom/View.h>
#include <Core/Thread/FutureValue.h>
#include <Core/Geom/GeomObj.h>

namespace SCIRun {
class Semaphore;
class GeometryData;


struct GeomReply {
  int portid;
  GeomReply();
  GeomReply(int);
};

class GeometryComm : public MessageBase {
public:
  GeometryComm(Mailbox<GeomReply> *);
  GeometryComm(int, GeomID, GeomHandle,
	       const string&, CrowdMonitor* lock);
  GeometryComm(int, LightID, LightHandle,
	       const string&, CrowdMonitor* lock);
  GeometryComm(int, GeomID);
  GeometryComm(int, LightID);
  GeometryComm(MessageTypes::MessageType, int);
  GeometryComm(MessageTypes::MessageType, int, Semaphore* wait);
  GeometryComm(MessageTypes::MessageType, int, int, View);
  GeometryComm(MessageTypes::MessageType, int portid,
	       FutureValue<GeometryData*>* reply,
	       int which_viewwindow, int datamask);
  GeometryComm(MessageTypes::MessageType, int portid,
	       FutureValue<int>* reply);
  GeometryComm(const GeometryComm &copy);
  virtual ~GeometryComm();

  Mailbox<GeomReply> *reply;
  int portno;
  GeomID serial;
  GeomHandle obj;
  LightID lserial;
  LightHandle light;
  string name;
  CrowdMonitor* lock;
  Semaphore* wait;
  View view;

  GeometryComm* next;

  int which_viewwindow;
  int datamask;
  FutureValue<GeometryData*>* datareply;
  FutureValue<int>* nreply;
};

} // End namespace SCIRun


#endif /* SCI_Datatypes_GeometryComm_h */
