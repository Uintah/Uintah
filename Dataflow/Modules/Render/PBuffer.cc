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
 *  PBuffer.cc: Render geometry to a pbuffer using opengl
 *
 *  Written by:
 *   Kurt Zimmerman and Milan Ikits
 *   Department of Computer Science
 *   University of Utah
 *   December 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Modules/Render/PBuffer.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;

namespace SCIRun {

PBuffer::PBuffer( int doubleBuffer ):
  width_(0), height_(0), colorBits_(8),
  doubleBuffer_(doubleBuffer),
  depthBits_(8),
  valid_(false),
  cx_(0)
{}

void
PBuffer::create(Display* dpy, int screen,
		int width, int height, int colorBits, int depthBits)
{
  dpy_ = dpy;
  screen_ = screen;
  width_ = width;
  height_ = height;
  colorBits_ = colorBits;
  
  // Set up a pbuffer associated with dpy
  int minor = 0 , major = 0;
  glXQueryVersion(dpy, &major, &minor);
  if( major >= 1 || (major == 1 &&  minor >1)) { // we can have a pbuffer
    //    cerr<<"We can have a pbuffer!\n";
    int attrib[32];
    int i = 0;
    attrib[i++] = GLX_RED_SIZE; attrib[i++] = colorBits;
    attrib[i++] = GLX_GREEN_SIZE; attrib[i++] = colorBits;
    attrib[i++] = GLX_BLUE_SIZE; attrib[i++] = colorBits;
    attrib[i++] = GLX_ALPHA_SIZE; attrib[i++] = colorBits;  
    attrib[i++] = GLX_DEPTH_SIZE; attrib[i++] = depthBits_;
    attrib[i++] = GLX_DRAWABLE_TYPE; attrib[i++] = GLX_PBUFFER_BIT;
    attrib[i++] = GLX_DOUBLEBUFFER; attrib[i++] = doubleBuffer_;
    attrib[i] = None;
    int nelements;
    fbc_ = glXChooseFBConfig( dpy, screen, attrib, &nelements );
    if( fbc_ == 0 ){
      //      cerr<<"Can not configure for Pbuffer\n";
      return;
    }

    i = 0;
    attrib[i++] = GLX_PBUFFER_WIDTH; attrib[i++] = width_;
    attrib[i++] = GLX_PBUFFER_HEIGHT; attrib[i++] = height_;
    attrib[i] = None;
    pbuffer_ = glXCreatePbuffer( dpy, *fbc_, attrib );
    if( pbuffer_ == 0 ) {
      //      cerr<<"Cannot create Pbuffer\n";
      return;
    }

    cx_ = glXCreateNewContext( dpy, *fbc_, GLX_RGBA_TYPE, NULL, True);
    if( !cx_ ){
      //      cerr<<"Cannot create Pbuffer context\n";
      return;
    }
    //    else cerr<<"Pbuffer successfully created\n";
    valid_ = true;
  } else {
    //    cerr<<"GLXVersion = "<<major<<"."<<minor<<"\n";
    cx_ = 0;
  }

}

void
PBuffer::destroy()
{
  if( cx_ ) {
    glXDestroyContext( dpy_, cx_ );
    cx_ = 0;
  }

  if( pbuffer_ ) {
    glXDestroyPbuffer( dpy_, pbuffer_);
    pbuffer_ = 0;
  }
  valid_ = false;
}

void
PBuffer::makeCurrent()
{
  if( valid_ )
    glXMakeCurrent( dpy_, pbuffer_, cx_ );
}

bool
PBuffer::is_current()
{
  if ( cx_ == glXGetCurrentContext() ){
    return true;
  } else {
    return false;
  }
}

} // end namespace SCIRun
