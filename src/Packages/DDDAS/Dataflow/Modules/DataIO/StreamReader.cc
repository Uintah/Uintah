//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : StreamReader.cc
//    Author : Martin Cole
//    Date   : Tue Aug 15 14:16:14 2006

  
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>



#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/DDDAS/share/share.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h> 
#include <Core/Thread/ConditionVariable.h>

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Util/Socket.h>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sys/types.h>

namespace DDDAS {

using namespace SCIRun;
class StreamReader;

class Listener: public Runnable
{
public:
  Listener(int port, StreamReader *mod) :
    mod_(mod),
    dead_(false)
  {
    if (! (sock_.create() && sock_.bind(port))) {
      dead_ = true;
    }
  }
  virtual ~Listener() {}
  
  virtual void run();
  void set_dead(bool p) { dead_ = p; }

private:
  StreamReader          *mod_;
  bool                  dead_;
  Socket                sock_;
};

class DDDASSHARE StreamReader : public Module {

public:
  //! Virtual interface
  StreamReader(GuiContext* ctx);

  virtual ~StreamReader();
  virtual void execute();
  void new_data_notify(const string fname, void *buf, size_t bytes);
private:
  void register_with_broker();

  //! GUI variables
  GuiString     brokerip_;
  GuiInt        brokerport_;
  GuiString     groupname_;
  GuiInt        listenport_;

  Listener     *listener_;
  Thread       *listener_thread_;
};

class DataHandler: public Runnable
{
public:
  DataHandler(StreamReader *mod, Socket *conn) :
    mod_(mod),
    conn_(conn)
  {}
  virtual ~DataHandler() 
  {
    if (conn_) { delete conn_; }
  }
  
  virtual void run();
private:
  StreamReader          *mod_;
  Socket                *conn_;
};

void
DataHandler::run()
{

  if (! conn_->is_valid()) {
    cerr << "bad socket: " << endl;
  }
  string fname;
  // get the string that represents the filename.
  if (! conn_->read(fname)) { return; }

  char bytes[8];
  
  // get the size of incoming data.
  if (conn_->read(&bytes, sizeof(long int)) != sizeof(long int)) {
    return;
  }
  char bytes1[8];
  bytes1[0] = bytes[7];
  bytes1[1] = bytes[6];
  bytes1[2] = bytes[5];
  bytes1[3] = bytes[4];
  bytes1[4] = bytes[3];
  bytes1[5] = bytes[2];
  bytes1[6] = bytes[1];
  bytes1[7] = bytes[0];

  long int *mylong;

//   mylong = (long int*)bytes;
//   cerr << "getting : " << *mylong << " bytes." << endl;

  mylong = (long int*)bytes1;
  cerr << "getting : " << *mylong << " bytes." << endl;

  char *buf = new char[*mylong];

  // get the rest of the data.
  if (conn_->read(buf, *mylong) != (int)*mylong) {
    return; 
  }

  // tell the module about the new data.
  mod_->new_data_notify(fname, buf, *mylong);
}




void
Listener::run()
{
  //! just accept and store connection sockets.
  cerr << "Listener thread started. Accepting con's from broker." << endl;
  while (!dead_ && sock_.listen()) {
    Socket *s = new Socket();
    if (sock_.accept(*s)) {
      mod_->remark("new connection accepted");
      // spawn the data handler thread, and go back to accepting.
      DataHandler *dh = scinew DataHandler(mod_, s);
      Thread *t = scinew Thread(dh, "StreamReader DataHandler thread.");
      t->detach();
    } else {
      dead_ = true;
    }
  }
}





DECLARE_MAKER(StreamReader);

StreamReader::StreamReader(GuiContext* ctx) : 
  Module("StreamReader", ctx, Source, "DataIO", "DDDAS"),
  brokerip_(get_ctx()->subVar("brokerip"), "localhost"),   
  brokerport_(get_ctx()->subVar("brokerport"), 8831),   
  groupname_(get_ctx()->subVar("groupname"), "wildfire"),   
  listenport_(get_ctx()->subVar("listenport"), 8835),
  listener_(0),
  listener_thread_(0)
{  
  cout << "(StreamReader::StreamReader) Inside" << endl;  

  if (! listener_) {
    int port = listenport_.get();
    listener_ = scinew Listener(port, this);
    listener_thread_ = scinew Thread(listener_, "StreamReader listener");
    listener_thread_->detach();
  }
  
  register_with_broker();
}


StreamReader::~StreamReader()
{
}


//! Blocks this thread until such time as accept returns the listener 
//! socket from the broker.
void
StreamReader::register_with_broker() 
{
  Socket sock;
  sock.create();

  if (! sock.connect(brokerip_.get(), brokerport_.get())) {
    error("connect failed");
  }

  ostringstream reg;
  reg << "register 127.0.0.1" << ":" << listenport_.get() << "\n";
  cerr << reg.str().c_str() << "............" << endl;
  if (! sock.write(reg.str().c_str())) {
    cerr << "error sending register" << endl;
  }

  ostringstream pass;
  pass << "password" << ": " << groupname_.get() << "\n";
  cerr << pass.str().c_str() << "............" << endl;
  if (! sock.write(pass.str().c_str())) {
    cerr << "error sending password" << endl;
  }

  cerr << "sent registration and password" << endl;
  
  cerr << "waiting for answer" << endl;
  string answer;
  sock.read(answer);
  cerr << "answer: " << answer << endl;
  if (answer != "SUCCEEDED") {
    error("registration with broker failed.");
  }
}

void 
StreamReader::execute()
{
  cout << "(StreamReader::execute) Inside" << endl;
}


void 
StreamReader::new_data_notify(const string fname, void *buf, size_t bytes)
{
  cerr << "got data, named: " << fname  << ", " << bytes 
       << " bytes long." << endl;
}

} // End namespace DDDAS




