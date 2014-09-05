/*
 *  ThetaView.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <tcl.h>
#include <tk.h>

#include <Core/Containers/Array2.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>
#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>

#include <Packages/MIT/share/share.h>

namespace MIT {

extern "C" Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

using namespace SCIRun;

class MITSHARE ThetaView : public Module {
public:
  ResultsHandle theta;

  // Gui
  GuiInt gui_params;
  GuiInt gui_select;
  GuiString gui_graph;
  
  // 
  Array2<double> stat;
  
  // OpenGL
  Window win;
  Display* dpy;
  GLXContext cx;
  GLuint base;  

  int xres, yres;

public:
  ThetaView(const string& id);

  virtual ~ThetaView();

  virtual void execute();
  
  virtual void tcl_command(TCLArgs&, void*);

private:
  void compute_statistics();
  void redraw();
  void plot();
  void make_raster_font();
  void print_string( char *);
};

extern "C" MITSHARE Module* make_ThetaView(const string& id) {
  return scinew ThetaView(id);
}

ThetaView::ThetaView(const string& id)
  : Module("ThetaView", id, Source, "Bayer", "MIT"),
    gui_params("params", id, this ),
    gui_select("select", id, this ),
    gui_graph("graph", id, this )
{
}

ThetaView::~ThetaView()
{
}

void 
ThetaView::execute()
{
  update_state(NeedData);

  ResultsIPort *input = (ResultsIPort *) get_iport("Posterior");
  input->get(theta);

  if ( !theta.get_rep() )
    return;

  update_state(JustStarted);

  compute_statistics();
  gui_params.set( theta->theta.size() );
}

void
ThetaView::compute_statistics()
{
  int nparams = theta->theta.size();
  int samples = theta->k.size(); 
  
  stat.newsize( nparams, 2 );

  for (int i=0; i<nparams; i++) {
    Array1<double> &values = theta->theta[i];
    stat(i,0) = stat(i,1) = values[0];
    for (int k=1; k<samples; k++) {
      if ( values[k] < stat(i,0) ) stat(i,0) = values[k];
      else if ( stat(i,1) < values[k] ) stat(i,1) = values[k];
    }
  }
}

void 
ThetaView::tcl_command(TCLArgs& args, void* userdata)
{
  if ( args[1] == "redraw" ) 
    redraw();
  else
    Module::tcl_command(args, userdata);

}

void
ThetaView::redraw()
{
  TCLTask::lock();

  Tk_Window tkwin=Tk_NameToWindow(the_interp,
				  const_cast<char *>(gui_graph.get().c_str()),
				  Tk_MainWindow(the_interp));
  if(!tkwin){
    cerr << "Unable to locate window!\n";
    TCLTask::unlock();
    return;
  }
  dpy=Tk_Display(tkwin);
  Window win=Tk_WindowId(tkwin);
  cx=OpenGLGetContext(the_interp, const_cast<char *>(gui_graph.get().c_str()));
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    TCLTask::unlock();
    return;
  }
  //    }
  
  // Get the window size
  xres=Tk_Width(tkwin);
  yres=Tk_Height(tkwin);

  if (!glXMakeCurrent(dpy, win, cx))
    cerr << "*glXMakeCurrent failed.\n";
  

  //make_raster_font();

  // Clear the screen...
  glViewport(0, 0, xres, yres);
  glClearColor(0.8, 0.8, 0.8, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  
//   glMatrixMode(GL_PROJECTION);
//   glLoadIdentity();
  //glOrtho(0, 150, 300, 0, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  
  plot();

  glXSwapBuffers( dpy, win);
  GLenum errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "Plot got an error from GL: " 
	 << (char*)gluErrorString(errcode) << endl;
  }
  glXMakeCurrent(dpy, None, NULL);
  TCLTask::unlock();
}

void
ThetaView::plot()
{
  if ( !theta.get_rep() || theta->k.size() == 0 )
    return;

  int n = 1; //gui_select.get();
  int last = theta->k.size()-1;

  glColor3f(0,0,0);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //glOrtho( 0, xres, 0, yres, -1, 1 );
  cerr << "ThetaView: " <<  theta->k[0]<<" " << theta->k[last]
       << "  "<< stat(n,0) << "   " <<  stat(n,1) << endl;
  glOrtho( theta->k[0], theta->k[last], stat(n,0), stat(n,1), -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // show axis
  glBegin(GL_LINES);

  // y axis
//   glVertex2i( 10, 10);
//   glVertex2i( 10, 200);
//   glVertex2i( 30, 10 );
//   glVertex2i( 200, 75);
  glVertex2f( theta->k[0], stat(n,0));
  glVertex2f( theta->k[0], stat(n,1));
  
  // x axis
  if ( stat(n,0) <= 0 && stat(n,1) >= 0) {
    glVertex2f( theta->k[0], stat(n,0));
    glVertex2f( theta->k[last], stat(n,1));
  }
   glEnd();

   glColor3f(0,0,.8);
   glBegin(GL_LINE_STRIP);
   for (int i=0; i<last; i++)
     glVertex2f( theta->k[i], theta->theta[1][i] );
   glEnd();

}

void 
ThetaView::make_raster_font()
{
  if (!cx)
    return;

  cerr << "Generating new raster font list!\n";
  XFontStruct *fontInfo;
  Font id;
  unsigned int first, last;
  Display *xdisplay;
  
  xdisplay = dpy;
  fontInfo = XLoadQueryFont(xdisplay, 
			    "-*-helvetica-medium-r-normal--12-*-*-*-p-67-iso8859-1");
  if (fontInfo == NULL) {
    printf ("no font found\n");
    exit (0);
  }
  
  id = fontInfo->fid;
  first = fontInfo->min_char_or_byte2;
  last = fontInfo->max_char_or_byte2;
  
  base = glGenLists((GLuint) last+1);
  if (base == 0) {
    printf ("out of display lists\n");
    exit (0);
  }
  glXUseXFont(id, first, last-first+1, base+first);
  /*    *height = fontInfo->ascent + fontInfo->descent;
   *width = fontInfo->max_bounds.width;  */
}

void 
ThetaView::print_string(char *s)
{
  glPushAttrib (GL_LIST_BIT);
  glListBase(base);
  glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
  glPopAttrib ();
}

} // End namespace MIT


