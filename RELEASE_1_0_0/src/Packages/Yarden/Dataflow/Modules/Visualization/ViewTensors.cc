/*
 *  ViewTensors.cc 
 *      View ViewTensors Tensor
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#include <stdio.h>
#include <unistd.h>

#include <Core/Thread/Time.h>
#include <Core/Containers/String.h>

#include <Core/Datatypes/ScalarFieldRG.h>

#include <Core/Thread/Thread.h>

#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>

#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>

#include <Packages/Yarden/Dataflow/Ports/TensorFieldPort.h>

#include <tcl.h>
#include <tk.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <values.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;


namespace Yarden {
using namespace SCIRun;
using namespace DaveW::Datatypes;
    
    
    // ViewTensors

    class ViewTensors : public Module 
    {
      // input
      TensorFieldIPort* in_tensor; // input tensor
      
      // tcl variables
      GuiInt tcl_slice;
      GuiInt tcl_mode;
      GuiInt tcl_nx, tcl_ny, tcl_nz;
      GuiInt tcl_num_slices;

      // Global variables
      TensorFieldHandle tensor_field;
      TensorField<SymTensor<float,3> > *t;

      Array3<unsigned char> tensor[6];
      int tensor_generation;
      int width, height;
      int mode;
      int slice;
      bool has_data;
      int nx, ny, nz;
      unsigned char *buffer;
      double min[6], max[6], factor[6];

      // GL Valriables
      bool opengl_initialized;
      GLXContext cx;
      Display*   dpy;
      Window     win;
      
    public:
      ViewTensors( const clString& id);
      virtual ~ViewTensors();
      
      virtual void execute();
      //virtual void do_execute();

    private:
      virtual void tcl_command(TCLArgs& args, void*); 
      template<class T>  void new_tensor( T *);

      int init_opengl();
      void display();
      void show( int i, int x, int y, int slice );
      void show_x( int i, int x, int y, int slice );
      void show_y( int i, int x, int y, int slice );
      void show_z( int i, int x, int y, int slice );
      void pre_display();
      void post_display();
      void reconfigure();
    };
    
    extern "C" Module* make_ViewTensors(const clString& id)
    {
      return scinew ViewTensors(id);
    }
    
    static clString module_name("ViewTensors");
    
    ViewTensors::ViewTensors(const clString& id)
      : Module("ViewTensors", id, Filter ),
	tcl_slice("slice", id, this ),
	tcl_mode("mode", id, this ),
	tcl_nx("nx", id, this ),
	tcl_ny("ny", id, this),
	tcl_nz("nz", id, this ),
	tcl_num_slices("num-slices", id, this )
    {
      // input ports
      in_tensor=scinew TensorFieldIPort(this, "Tensor Field",
				       TensorFieldIPort::Atomic);
      add_iport(in_tensor);

      tensor_generation = -1;
      width = 512;
      height = 700;
      mode = 0;
      opengl_initialized = false;
      has_data = false;
      buffer = 0;
    }

    ViewTensors::~ViewTensors()
    {
    }

    void
    ViewTensors::execute()
    {
      TensorFieldHandle field;
      if ( !in_tensor->get( field ) ) {
	error("no input tensor");
	return;
      }

      cerr << "get \n";
      if ( field->generation != tensor_generation ) {
	TensorFieldBase *base = field.get_rep();
	if ( !base ) {
	  cerr << "No base ?\n";
	  return;
	}
	
	TensorField< SymTensor<float,3> > *t = 
	  dynamic_cast<TensorField<SymTensor<float,3> >* >(field.get_rep());
	
 	if ( !t ) {
 	  cerr << "Not SymTensor<float,3> ?\n";
 	  return;
	}
	
	tensor_field = field;
  	new_tensor( t );
      }
      
      display();
    }    
    

    template<class T>
    void
    ViewTensors::new_tensor( T *input)
    {
      t = input;
      nx = input->dim3();
      ny = input->dim2();
      nz = input->dim1();

//       tcl_nx.set(nx-1);
//       tcl_ny.set(ny-1);
//       tcl_nz.set(nz-1);

      ostringstream str;
      str << id << " new-info " << nx << " " << ny << " " << nz;
      TCL::execute(str.str().c_str());

      input->get_minmax( min, max );

      for (int i=0; i<6; i++) 
	factor[i] = 255.0/(max[i]-min[i]);

      if ( buffer ) delete buffer;
      int size;
      if ( nx > ny ) 
	size = nx * (ny > nz ? ny : nz);
      else 
	size = ny * (nx > nz ? nx : nz);
      buffer = scinew unsigned char[size];

      has_data = true;
    }

    void
    ViewTensors::display()
    {
      int x,y;   

      if ( !opengl_initialized || !has_data )
	return;
      
      pre_display();
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);

      reset_vars();
      mode = tcl_mode.get();
      //      cerr << "mode = " << mode << endl;
      switch (mode) {
      case 0:
	x = 10;
	y = 10+nz;
	show_x( 0, x, y, slice );
	x += ny+10;
	show_x( 1, x, y, slice );
	x += ny+10;
	show_x( 2, x, y, slice );
	x = ny+20;
	y += nz+10;
	show_x( 3, x, y, slice );
	x += ny+10;
	show_x( 4, x, y, slice );
	y += nz+10;
	show_x( 5, x, y, slice );
	break;
      case 1:
	x = 10;
	y = 10+nz;
	show_y( 0, x, y, slice );
	x += nx+10;
	show_y( 1, x, y, slice );
	x += nx+10;
	show_y( 2, x, y, slice );
	x = nx+20;
	y += nz+10;
	show_y( 3, x, y, slice );
	x += nx+10;
	show_y( 4, x, y, slice );
	y += nz+10;
	show_y( 5, x, y, slice );
	break;
      case 2:
	x = 10;
	y = 10+ny;
	show_z( 0, x, y, slice );
	x += nx+10;
	show_z( 1, x, y, slice );
	x += nx+10;
	show_z( 2, x, y, slice );
	x = nx+20;
	y += ny+10;
	show_z( 3, x, y, slice );
	x += nx+10;
	show_z( 4, x, y, slice );
	y += ny+10;
	show_z( 5, x, y, slice );
	break;
      }
      post_display();
    }

    void 
    ViewTensors::show_x( int i, int at_x, int at_y, int slice )
    {
      glRasterPos2i(at_x,at_y);
      
      int p = 0;
      for (int z=nz-1; z>=0; z--)
	for (int y=0; y<ny; y++)
	  buffer[p++] = (unsigned char)
	    ((t->data(z,y,slice)[i]-min[i]) *factor[i]);
      
      glDrawPixels( ny, nz, 
		    GL_LUMINANCE, GL_UNSIGNED_BYTE, 
		    buffer );
      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR)
	cerr << "draw: "<< (char*)gluErrorString(errcode)<< endl;
    }      
    
    void 
    ViewTensors::show_y( int i, int at_x, int at_y, int slice )
    {
      glRasterPos2i(at_x,at_y);
      
      int p = 0;
      for (int z=nz-1; z>=0; z--)
	for (int x=0; x<nx; x++)
	  buffer[p++] = (unsigned char )
	    ((t->data(z,slice,x)[i]-min[i]) *factor[i]);
      
      glDrawPixels( nx, nz, 
		    GL_LUMINANCE, GL_UNSIGNED_BYTE, 
		    buffer );
      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR)
	cerr << "draw: "<< (char*)gluErrorString(errcode)<< endl;
    }      

    void 
    ViewTensors::show_z( int i, int at_x, int at_y, int slice )
    {
      glRasterPos2i(at_x,at_y);
      
      int p = 0;
      for (int y=ny-1; y>=0; y--)
	for (int x=0; x<nx; x++)
	  buffer[p++] = (unsigned char )
	    ((t->data(slice,y,x)[i]-min[i]) *factor[i]);
      
      glDrawPixels( nx, ny, 
		    GL_LUMINANCE, GL_UNSIGNED_BYTE, 
		    buffer );
      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR)
	cerr << "draw: "<< (char*)gluErrorString(errcode)<< endl;
    }      

    void ViewTensors::tcl_command(TCLArgs& args, void* userdata) 
    {
      if(args[1] == "slice") {
	args[2].get_int( slice );
	display();
      }
      else if(args[1] == "redraw_all") {
	if ( !opengl_initialized ) 
	  init_opengl();
	display();
      }
      else if(args[1] == "configure") {
	args[2].get_int(width);
	args[3].get_int(height);
 	cerr << width << " x " << height << endl;
	reconfigure();
	display();
      }
      else {
	Module::tcl_command(args, userdata);
      }
    }
    
    int
    ViewTensors::init_opengl()
    {
      TCLTask::lock();

      cerr << "in init " << width << " " << height << endl;
      clString myname(clString(".ui")+id+".gl.gl");
      
      Tk_Window tkwin=Tk_NameToWindow(the_interp,
			    const_cast<char *>(myname()),
			    Tk_MainWindow(the_interp));
      
      if(!tkwin) {
	cerr << "Unable to locate window!\n";
	
	// unlock the mutex
	TCLTask::unlock();
	return 0;
      }
      

      dpy=Tk_Display(tkwin);
      win=Tk_WindowId(tkwin);

      cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
      if(!cx) {
	cerr << "Unable to create OpenGL Context!\n";
	TCLTask::unlock();
	return 0;
      }

      TCLTask::unlock();

      opengl_initialized = true;
      reconfigure();
      return 1;
    }

    void
    ViewTensors::reconfigure()
    {
      if ( !opengl_initialized )
	return;

      TCLTask::lock();

      glXMakeCurrent(dpy, win, cx);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0, width-1, height-1, 0, -1, 1);
      glViewport(0, 0, width, height);

      glDrawBuffer(GL_FRONT);
      glClearColor( 0, 0, 0, 0 );
      glClear(GL_COLOR_BUFFER_BIT);

      glXMakeCurrent(dpy, None, NULL);

      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR)
	cerr << "post error: "<< (char*)gluErrorString(errcode)<< endl;

      TCLTask::unlock();
    }

    void
    ViewTensors::pre_display()
    {
      TCLTask::lock();

      glXMakeCurrent(dpy, win, cx);

      glDrawBuffer(GL_BACK);

      glClearColor( 0, 0, 0, 0 );
      glClear(GL_COLOR_BUFFER_BIT);

      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR)
	cerr << "pre error: "<< (char*)gluErrorString(errcode)<< endl;
    }
      
    void
    ViewTensors::post_display()
    {
      glXSwapBuffers(dpy,win);
      glXMakeCurrent(dpy, None, NULL);
      GLenum errcode;
      while((errcode=glGetError()) != GL_NO_ERROR)
	cerr << "post error: "<< (char*)gluErrorString(errcode)<< endl;
      TCLTask::unlock();
    }
} // End namespace Yarden

