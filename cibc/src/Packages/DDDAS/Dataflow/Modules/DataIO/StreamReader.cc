//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Util/Socket.h>

#include <geotiff.h>
#include <geo_normalize.h>
#include <geovalues.h>
#include <tiffio.h>
#include <xtiffio.h>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sys/types.h>

namespace DDDAS {

using namespace SCIRun;

typedef ImageMesh<QuadBilinearLgn<Point> >                   IMesh;
typedef QuadBilinearLgn<Vector>                              DBasisrgb;
typedef GenericField<IMesh, DBasisrgb, FData2d<Vector, IMesh> > IFieldrgb;
typedef QuadBilinearLgn<double>                              DBasisgs;
typedef GenericField<IMesh, DBasisgs, FData2d<double, IMesh> > IFieldgs;


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
  bool register_with_broker();
  BBox get_bounds(GTIF *gtif, GTIFDefn *defn,
		  int xsize, int ysize);

  template <class Fld>
  bool fill_image(Fld *fld, IMesh *im, int bpp, int spp, uint32 *buf);

  //! GUI variables
  GuiString     brokerip_;
  GuiInt        brokerport_;
  GuiString     groupname_;
  GuiInt        listenport_;

  Listener     *listener_;
  Thread       *listener_thread_;
  bool          registered_;
  FieldHandle   out_fld_h_;
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

  //char bytes[8];
  
  uint32_t bytes;

  // get the size of incoming data.
  if (conn_->read(&bytes, sizeof(uint32_t)) != sizeof(uint32_t)) {
    return;
  }
  // convert from network byte order to host byte order.
  bytes = ntohl(bytes);

  cerr << "getting : " << bytes << " bytes." << endl;
  char *buf = new char[bytes];

  // get the rest of the data.
  if (conn_->read(buf, bytes) != (int)bytes) {
    cerr << "Error did not read "<< bytes << " bytes." << endl;
    cerr << "DataHandler exiting...." << endl;
    return; 
  }

  // tell the module about the new data.
  mod_->new_data_notify(fname, buf, bytes);
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
      cerr << "Listener thread exiting." << endl;
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
  listener_thread_(0),
  registered_(false),
  out_fld_h_(0)
{  
  cout << "(StreamReader::StreamReader) Inside" << endl;  

  if (! listener_) {
    int port = listenport_.get();
    listener_ = scinew Listener(port, this);
    listener_thread_ = scinew Thread(listener_, "StreamReader listener");
    listener_thread_->detach();
  }
  
  //registered_ = register_with_broker();
}


StreamReader::~StreamReader()
{
}


//! Blocks this thread until such time as accept returns the listener 
//! socket from the broker.
bool
StreamReader::register_with_broker() 
{
  Socket sock;
  sock.create();

  if (! sock.connect(brokerip_.get(), brokerport_.get())) {
    error("connect failed");
    return false;
  }

  ostringstream reg;
  reg << "register " << Socket::get_local_ip() << ":" 
      << listenport_.get() << "\n";
  cerr << reg.str().c_str() << "............" << endl;
  if (! sock.write(reg.str().c_str())) {
    error("error sending register");
    return false;
  }

  ostringstream pass;
  pass << "password" << ": " << groupname_.get() << "\n";
  cerr << pass.str().c_str() << "............" << endl;
  if (! sock.write(pass.str().c_str())) {
    error("error sending password");
    return false;
  }

  remark("sent registration and password, waiting for answer.");
  
  string answer;
  sock.read(answer);
  cerr << "answer: " << answer << endl;
  if (answer != "SUCCEEDED\n") {
    error("registration with broker failed.");
    return false;
  }
  return true;
}

void 
StreamReader::execute()
{
  if (! registered_) {
    registered_ = register_with_broker();
    if (! registered_) {
      error("Registration with broker failed. returning...");
      return;
    }
  }
  cout << "(StreamReader::execute) Registered with broker." << endl;

  if (out_fld_h_.get_rep()) {
    send_output_handle("Output Sample Field", out_fld_h_);
    out_fld_h_ = 0;
  }
}

Vector
get_value(uint32* buf, unsigned int idx, int bpp, int spp) 
{  
  //if (spp == 1 && bpp == 8) {
    uint32 p = buf[idx];
    unsigned char r = TIFFGetR(p);
    unsigned char g = TIFFGetG(p);
    unsigned char b = TIFFGetB(p);
    //cerr << "rgb: " << (int)r << ", " << (int)g << ", " << (int)b << endl;
    return Vector(r / 255., g / 255., b / 255.);
    //}

  cerr << "WARNING: default get_value is 0" << endl;
  return Vector(0,0,0);
}

template <class Fld>
bool 
StreamReader::fill_image(Fld *fld, IMesh *im, int bpp, int spp, uint32 *buf)
{
  IMesh::Node::iterator iter, end;
  im->begin(iter);
  im->end(end);
  
  unsigned int idx = 0;
  while (iter != end) {
    typename Fld::value_type val = get_value(buf, idx, bpp, spp);
    IMesh::Node::index_type ni = *iter;
    fld->set_value(val, ni);

    ++iter;
    ++idx;
  }
  
  return true;
}



void 
StreamReader::new_data_notify(const string fname, void *buf, size_t bytes)
{
  cerr << "got data, named: " << fname  << ", " << bytes 
       << " bytes long." << endl;

  if (fname.find(".lgo") != string::npos) {
    // handle the header file
    string header(bytes, '\0');
    char *c = (char*)buf;
    for (unsigned int i = 0; i < bytes; i++) {
      header[i] = *c;
      c++;
    }
    
    for (unsigned int i = 0; i < bytes; i++) {
      cerr << header[i];
    }
    cerr << endl;
  }

  if (fname.find(".tif") != string::npos) {
    const char* tmpfn = "/tmp/dddas.tif";
    FILE *fd = fopen(tmpfn, "w");
    size_t status = fwrite(buf, sizeof(char), bytes, fd);
    fclose(fd);

    TIFF* tif = XTIFFOpen(tmpfn, "r");
    if (!tif) return;

    GTIF* gtif = GTIFNew(tif);
    GTIFDefn	defn;

    if(GTIFGetDefn(gtif, &defn))
    {
      int xsize, ysize;
      uint16 spp, bpp, photo;

      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &xsize);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &ysize);
      TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp);
      TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
      TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
//       cerr << "tiff size: (" << xsize << ", " << ysize << ")" << endl; 
//       cerr << "bits/pixel: " << bpp << endl;
//       cerr << "samples/pixel: " << spp << endl;
//       cerr << "photo: " << photo << endl;


      int npixels = xsize * ysize;
      uint32* raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
      if (raster != NULL) {
	if (TIFFReadRGBAImage(tif, xsize, ysize, raster, 0)) {
	  BBox bb = get_bounds(gtif, &defn, xsize, ysize);

	  IMesh* im = new IMesh(xsize, ysize, bb.min(), bb.max());
	  BBox cbb = im->get_bounding_box();

	  IFieldrgb *ifld = new IFieldrgb(im);
	  fill_image(ifld, im, bpp, spp, raster);
	  out_fld_h_ = ifld;
	  want_to_execute();
	} else {
	  cerr << "could not read image" << endl;
	  return;
	}
	//_TIFFfree(raster);
      }
      TIFFClose(tif);

    }
 
    cerr << endl << "recieved data: " << fname << endl;

  }
}


BBox 
StreamReader::get_bounds(GTIF *gtif, GTIFDefn *defn,
			 int xsize, int ysize) 
{
  double x = 0.0;
  double y = 0.0;
  double tx = x;
  double ty = y;

  cerr << "Corner Coordinates:" << endl;
  if(!GTIFImageToPCS(gtif, &tx, &ty)) 
  {
    cerr << "unable to transform points between pixel/line and PCS space"
	 << endl;
    //return BBox();
  }
  BBox bb;

  tx = 0.0;
  ty = ysize;
  GTIFImageToPCS(gtif, &tx, &ty);
  cerr << "LL" << tx << ", " << ty << endl;
  bb.extend(Point(tx, ty, 0.0));

  tx = xsize;
  ty = 0.0;
  GTIFImageToPCS(gtif, &tx, &ty);
  cerr << "UR" << tx << ", " << ty << endl;
  bb.extend(Point(tx, ty, 0.0));

  return bb;
}

} // End namespace DDDAS




