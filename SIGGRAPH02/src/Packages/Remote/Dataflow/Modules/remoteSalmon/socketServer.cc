

/*
 *  socketServer.cc:
 *
 *  Written by:
 *   David Hart
 *   Department of Computer Science
 *   University of Utah
 *   May 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Remote/Dataflow/Modules/remoteSalmon/socketServer.h>

namespace Remote {
//----------------------------------------------------------------------
socketServer::socketServer(OpenGLServer* opengl): opengl(opengl) {
  
				// start the remote Communication
  cerr << "socketServer: starting up comm server" << endl;
  listensock = new Socket;
  int port;
  char machine[80];
  serverRunning = true;

  if (gethostname(machine, 79) == -1) {
    cerr << "socketServer warning: can't read local host name" << endl;
  }
    
				// try to grab a server port
  for (port = VR_PORT; port < VR_PORT + VR_NUMPORTS; port++) {
    cout << "socketServer: tyring listen on port " << port << endl;
    if (listensock->ListenTo(machine, port)) break;
    listensock->Reset();
  }

				// if we couldn't grab a server port,
				// bail out
  if (!(listensock->isConnected())) {
    cerr << "socketServer error: couldn't listen -- \n"
	 << " -- the comm server was not started" << endl;
    serverRunning = false;
  }

  //listensock->Block(0);
  
  socks.push_back(listensock);

}

//----------------------------------------------------------------------
socketServer::~socketServer() {
  int i;
  
  cerr << "socketServer: shutting down comm server" << endl;
  
				// shut down the sockets
  for (i = 0; i < socks.size(); i++) {
    //socks[i]->Write(VR_QUIT);
    socks[i]->Close();
    delete socks[i];
  }
  
  //listensock->Close();
  //delete listensock;
}

//----------------------------------------------------------------------
// add a new socket to the list of open connections
void
socketServer::addConnection(Socket* sock) {
  socks.push_back(sock);
}

//----------------------------------------------------------------------
// shutdown and remove a socket from the list of open connections
void
socketServer::removeConnection(Socket* sock) {
  int i;

				// sanity check
  if (socks.empty()) return;

				// look for socket to remove
  for (i = 0; i < socks.size(); i++) {
    if (socks[i] == sock) break;
  }

				// if we didn't find it, bail
  if (i == socks.size()) return;

				// close the socket and free its
				// memory
  sock->Close();
  delete sock;
    
				// remove it my moving up everyone
				// else
  for (; i < socks.size()-1; i++) {
    socks[i] = socks[i+1];
  }

				// shrink the array
  socks.pop_back();
}


//----------------------------------------------------------------------
void spin() {
  /*
  static int pos = 0;
  static char* spinner = "-\\|/";
  pos = (pos + 1) % strlen(spinner);
  cerr << "     \r" << spinner[pos] << endl;
  */
  cerr << "." << flush;
}

//----------------------------------------------------------------------
void
socketServer::run() {
  
  Socket* sock;
  vector<Socket*> deletesocks;
  int msg;
  int i, err;
  float vect[3];

  cerr << "socketServer: waiting for connection" << endl;
  

  while (serverRunning) {

				// heartbeat
    spin();

    i = Socket::FindReadyToRead(&socks[0], socks.size(), true);
    sock = socks[i];

    				// check for an incoming call
    if (i == 0) {
    
				// answer the call
      cerr << "socketServer: accepting connection" << endl;
      sock = listensock->AcceptConnection();
      if (sock) {
	//sock->Block(0);
	addConnection(sock);
	cerr << "socketServer: got connection" << endl;
	//sock->Write(VR_SETPOS);
	//sendView(sock);
	//sock->Write(VR_ENDMESSAGE);
      }
      
    }
    else {

				// try to service incoming requests
      cerr << "socketServer: reading" << endl;
      err = sock->Read(msg);
      if (err == SOCKET_ERROR || !sock->isConnected()) { // error
	cerr << "socketServer: lost connection" << endl;
	deletesocks.push_back(sock);
      }
      else if (err > 0) {	// good message - ready to go
	switch(msg) {
				// client requested the camera
				// information
	case VR_GETPOS: {
	  DEBUG(VR_GETPOS);
	  sock->Write(VR_SETPOS);
	  sendView(sock);
	  sock->Write(VR_ENDMESSAGE);
	  
	}
	break;
				// client requested the camera to move to the
				// given viewing position/direction
	case VR_SETPOS: {
	  DEBUG(VR_SETPOS);
	  receiveView(sock);
	}
	break;
				// client requested zbuffer and image
				// data of the current view
	case VR_GETSCENE: {
	  DEBUG(VR_GETSCENE);
	  receiveView(sock);
				// send the current view info
	  sock->Write(VR_SETSCENE);
	  sendView(sock);
	  sendZBuffer(sock);
	  sock->Write(VR_ENDMESSAGE);	  
	}
	break;

	case VR_GETGEOM: {
	  DEBUG(VR_GETSCENE);
	  receiveView(sock);
				// send the current view info
	  sock->Write(VR_SETGEOM);
	  sendView(sock);
	  sendGeom(sock);
	  sock->Write(VR_ENDMESSAGE);	  
	}
	break;

				// client requested that the server
				// accept and display a new view -
				// this wont be supported and will be
				// ignored for now
	case VR_SETSCENE: {

	  DEBUG(VR_SETSCENE);
	  sock->Read(vect, 3);
	  cerr << "pos: " << vect[0] << " " << vect[1]
	       << " " << vect[2] << endl;
	  sock->Read(vect, 3);
	  cerr << "dir: " << vect[0] << " " << vect[1]
	       << " " << vect[2] << endl;
	  sock->Read(vect, 3);
	  cerr << "up: " << vect[0] << " " << vect[1]
	       << " " << vect[2] << endl;
	  
	}
	break;
	
				// client requested the connection to
				// be terminated
	case VR_QUIT: {
	  DEBUG(VR_QUIT);
	  cerr << "socketServer: closing connection" << endl;
	  deletesocks.push_back(socks[i]);
	}
	break;
	
	}

	err = sock->Read(msg);
	if (err == SOCKET_ERROR || !sock->isConnected()) { // error
	  cerr << "socketServer: lost connection" << endl;
	  deletesocks.push_back(sock);
	}
	else if (msg != VR_ENDMESSAGE) {
	  cerr << "socketServer: out of synch - closing connection"
	       << endl;
	  deletesocks.push_back(sock);
	}
	
      }
    }
  
				// shutdown and free all the
				// connections we lost
    for (i = 0; i < deletesocks.size(); i++) {
      removeConnection(deletesocks[i]);
    }
    
  }
    
}


//----------------------------------------------------------------------
void
socketServer::sendView(Socket* sock) {
  
  View tmpview(opengl->roe->view.get());
	  
  //Vector gaze = tmpview.lookat() - tmpview.eyep();
  //gaze.normalize();

  
	  
  sock->Write(float(tmpview.eyep().x()));
  sock->Write(float(tmpview.eyep().y()));
  sock->Write(float(tmpview.eyep().z()));

  DEBUG(tmpview.eyep());
	  
  sock->Write(float(tmpview.lookat().x()));
  sock->Write(float(tmpview.lookat().y()));
  sock->Write(float(tmpview.lookat().z()));

  //sock->Write(float(gaze.x()));
  //sock->Write(float(gaze.y()));
  //sock->Write(float(gaze.z()));
	  
  DEBUG(tmpview.lookat());
	  
  sock->Write(float(tmpview.up().x()));
  sock->Write(float(tmpview.up().y()));
  sock->Write(float(tmpview.up().z()));
	  
  DEBUG(tmpview.up());

  DEBUG(tmpview.fov());
	    
}

//----------------------------------------------------------------------
void
socketServer::receiveView(Socket* sock) {
  View tmpview;
  float vect[3];

  
				// read the position
  sock->Read(vect, 3);
  tmpview.eyep(Point(vect[0], vect[1], vect[2]));

				// read the lookat point
  sock->Read(vect, 3);
  tmpview.lookat(Point(vect[0], vect[1], vect[2]));

				// read the up vector
  sock->Read(vect, 3);
  tmpview.up(Vector(vect[0], vect[1], vect[2]));

				// get previous fov
  tmpview.fov(opengl->roe->view.get().fov());

  opengl->roe->setView(tmpview);
  
				// force a redraw
  opengl->roe->force_redraw();
  opengl->roe->redraw_if_needed();
	  
}

//----------------------------------------------------------------------
// send the visible objects in the window over the network as explicit
// geometry
void
socketServer::sendMesh(Socket* sock) {
  cerr << "REMOTEVIZ: sending a mesh (not)" << endl;
}

//----------------------------------------------------------------------
// grab the z buffer, compress it and send it over the network
void
socketServer::sendZBuffer(Socket* sock) {

  cerr << "Roe::sendZbuffer: sending the zbuffer" << endl;

  int i, j;
  int x, y;

  static char* img_rle=NULL;
  static int img_rle_datalenR;
  static int img_rle_datalenG;
  static int img_rle_datalenB;
  
  static int rle_len = 0;
  
  int Width = opengl->xres;
  int Height = opengl->yres;
  
  ZImage ZBuf(Width, Height);
  Image  Img(Width, Height);


  DEBUG("before redraw");
				// force a redraw so the view in the
				// back buffer is current.
  opengl->roe->force_redraw();
  opengl->roe->redraw_if_needed();
  
  DEBUG("after redraw");

  cerr << "socketServer::sendZbuffer: Extracting buffers" << endl;
    
  //glReadPixels(0, 0, ZBuf.wid, ZBuf.hgt, GL_DEPTH_COMPONENT,
  //GL_UNSIGNED_INT, (GLvoid *)ZBuf.Pix);

				// request the depth and color buffers
				// from the rendering window
  FutureValue<GeometryData*> result("socketServer::sendzbuffer");
  opengl->getData(GEOM_COLORBUFFER | GEOM_DEPTHBUFFER, &result);

  cerr << "waiting for result" << endl;
  GeometryData* res = result.receive();
  cerr << "got result" << endl;

  if (!res->colorbuffer || !res->depthbuffer) {
    cerr << "socketServer::sendZbuffer: missing color or depth buffer"
	 << endl;
  }

				// copy the result data into my own
				// buffers
  Color c;
  for (y = 0; y < Height; y++) {
    for (x = 0; x < Width; x++) {
      c = res->colorbuffer->get_pixel(x, y);
      Img(x, y).r = char(255.0 * CLAMP(c.r(), 0.0, 1.0));
      Img(x, y).g = char(255.0 * CLAMP(c.g(), 0.0, 1.0));
      Img(x, y).b = char(255.0 * CLAMP(c.b(), 0.0, 1.0));      
      //ZBuf(x, y) = (unsigned int)(res->depthbuffer->get_depth(x, y)
      //* (4294967295./1.));
      //ZBuf(x, y) = ((unsigned int *)(res->depthbuffer))[x + y*Width];
    }
  }

  memcpy(ZBuf.IPix(), res->depthbuffer, Width*Height*sizeof(unsigned int));
  delete [] (unsigned int *)res->depthbuffer;

  //ZBuf.SavePZM("zbuf.pzm");

  if (rle_len < Width * Height) {
    delete [] img_rle;
    rle_len = Width * Height;
    img_rle = new char[rle_len * 3];
  }
  
  //Img.Resize(256,256);
  
  cout << "socketServer::sendZbuffer: run length encoding image" << endl;
  
  img_rle_datalenR =
    RLE_ENCODE(Img.Pix, Img.wid * Img.hgt, (unsigned char*)img_rle,
      (unsigned char)0, (unsigned char)1, 255, 3); 
  img_rle_datalenG =
    RLE_ENCODE(Img.Pix+1, Img.wid * Img.hgt,
      (unsigned char*)img_rle + img_rle_datalenR,
      (unsigned char)0, (unsigned char)1, 255, 3);
  img_rle_datalenB =
    RLE_ENCODE(Img.Pix+2, Img.wid * Img.hgt,
      (unsigned char*)img_rle + img_rle_datalenR + img_rle_datalenG,
      (unsigned char)0, (unsigned char)1, 255, 3);
  
  cout << "socketServer::sendZbuffer: image is compressed to " <<
    (float)(img_rle_datalenR+img_rle_datalenG+img_rle_datalenB) /
    (3 * rle_len) << "%" << endl;
  
  cout << "socketServer::sendZbuffer: sending image" << endl;


				// send the scene info
  sock->Write(Img.wid);
  sock->Write(Img.hgt);
  
  //sock->Write(&globalTranslate.x, 3);
  //sock->Write(globalScale);
  
  sock->Write(img_rle_datalenR);
  sock->Write(img_rle_datalenG);
  sock->Write(img_rle_datalenB);
  sock->Write((char*)img_rle, img_rle_datalenR +
    img_rle_datalenG + img_rle_datalenB);
  
  //------------------------------
  // simplify and send zbuffer as a mesh
  
  cout << "socketServer::sendZbuffer: simplifying zbuffer" << endl;
  
  HeightSimp* HFSimplifier = new HeightSimp(ZBuf, 0x01000000, 0x01000000);

  //Timer T;
  //T.Start();

  cout << "socketServer::sendZbuffer: simplifying zbuf mesh" << endl;
    
  SimpMesh *SM = HFSimplifier->HFSimp();
    
  delete HFSimplifier;

  //cerr << "After HFSimp = " << T.Read() << endl;

  /*
  //if (Simplify) {
      
    int TargetTris = SM->FaceCount / 4;
    double BScale = 2.1;
    
    SM->Dump();
    SM->Simplify(TargetTris, BScale);
      
    //cerr << "After Simp = " << T.Read() << endl;
      
    SM->FixFacing();
    SM->Dump();
      
    //}
    */

  Model* Mo = new Model(SM->ExportObject());

  delete SM;

  //cerr << "After Export = " << T.Read() << endl;

  cerr << "socketServer::sendZbuffer: transforming model" << endl;
    
  double modelmat[16];
  double projmat[16];
  int viewport[4];

  memset(modelmat, 0, 16*sizeof(double));
  memset(projmat, 0, 16*sizeof(double));
  memset(viewport, 0, 4*sizeof(int));
  
  Vector newverts;
  Vector* oldverts;
  
  opengl->send_mb.send(DO_GETGLSTATE);
  opengl->send_mb.send(GL_MODELVIEW_MATRIX);
  opengl->gl_in_mb.send((void*)modelmat);
  if (opengl->gl_out_mb.receive() != (void*)modelmat) {
    cerr << "woops1" << endl;
  }

  opengl->send_mb.send(DO_GETGLSTATE);
  opengl->send_mb.send(GL_PROJECTION_MATRIX);
  opengl->gl_in_mb.send((void*)projmat);
  if (opengl->gl_out_mb.receive() != (void*)projmat) {
    cerr << "woops2" << endl;
  }

  opengl->send_mb.send(DO_GETGLSTATE);
  opengl->send_mb.send(GL_VIEWPORT);
  opengl->gl_in_mb.send((void*)viewport);
  if (opengl->gl_out_mb.receive() != (void*)viewport) {
    cerr << "woops3" << endl;
  }

  //Mo->Flatten();
  
  cerr << "reprojecting" << endl;

  for (i = 0; i < Mo->Objs.size(); i++) {
    for (j = 0; j < Mo->Objs[i].verts.size(); j++) {
      oldverts = &(Mo->Objs[i].verts[j]);
      Mo->Objs[i].texcoords.push_back
	(Vector(oldverts->x/(double)Img.wid,
	  oldverts->y/(double)Img.hgt, 0));
      if (gluUnProject(oldverts->x, oldverts->y, oldverts->z,
	modelmat, projmat, viewport,
	&newverts.x, &newverts.y, &newverts.z) == GL_FALSE) {
	DEBUG(*oldverts);
	DEBUG(GL_FALSE);
      }
      else {
	//cerr << *oldverts << " -> " << newverts << endl;
      }
      *oldverts = newverts;
    }
  }

  //cerr << "After Transform = " << T.Read() << endl;

  cerr << "removing triangles" << endl;
  View tmpview(opengl->roe->view.get());
  Vector pos(tmpview.eyep().x(), tmpview.eyep().y(), tmpview.eyep().z());
  Mo->RemoveTriangles(pos, 0.2);
    
  cerr << "sending" << endl;
  sendModel(*Mo, sock);
  delete Mo;

  //cerr << "After Send = " << T.Read() << endl;
  //cerr << "BBox = " << Mo->Box << endl;
    
  cerr << "socketServer::sendZbuffer: done writing" << endl;
  
}

//----------------------------------------------------------------------
void
socketServer::sendGeom(Socket* sock) 
{
  FutureValue<GeometryData*> result("socketServer::sendgeom");
  opengl->getData(GEOM_TRIANGLES, &result);

  cerr << "waiting for result" << endl;
  GeometryData* res = result.receive();
  cerr << "got result" << endl;

  if ( !res->depthbuffer) {
    cerr << "socketServer::sendGeom: missing triangles array"
	 << endl;
  }

  Array1<float> &t =  *(Array1<float> *)res->depthbuffer;

  cerr << "send Geom: " << t.size()/3 << "polys" << endl;

  Object *obj= new Object;
  obj->verts.reserve(t.size()/3);
  for (int i=0; i<t.size(); i+=3)
    obj->verts.push_back( Vector(t[i], t[i+1], t[i+2]) );
  Model* Mo = new Model(*obj);

  cerr << "sending" << endl;
  sendModel(*Mo, sock);
  delete Mo;

  delete &t;
  cerr << "socketServer::sendGeom: done writing" << endl;
}

} // End namespace Remote



