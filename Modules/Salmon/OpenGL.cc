
#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Roe.h>
#include <Modules/Salmon/Salmon.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Geom/Geom.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/RenderMode.h>
#include <Geom/Light.h>
#include <Math/Trig.h>
#include <TCL/TCLTask.h>
#include <tcl/tcl7.3/tcl.h>
#include <tcl/tk3.6/tk.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <strstream.h>

const int STRINGSIZE=200;

class OpenGL : public Renderer {
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    clString myname;
    char* strbuf;
    int maxlights;
public:
    OpenGL();
    virtual ~OpenGL();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
    virtual void get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&);
    virtual void hide();
    virtual void put_scanline(int y, int width, Color* scanline, int repeat=1);
};

static OpenGL* current_drawer=0;
static const int pick_buffer_size = 512;
static const double pick_window = 5.0;

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;

static Renderer* make_OpenGL()
{
    return new OpenGL;
}

static int query_OpenGL()
{
    TCLTask::lock();
    int have_opengl=glXQueryExtension(Tk_Display(Tk_MainWindow(the_interp)),
				      NULL, NULL);
    TCLTask::unlock();
    return have_opengl;
}

RegisterRenderer OpenGL_renderer("OpenGL", &query_OpenGL, &make_OpenGL);

OpenGL::OpenGL()
: tkwin(0)
{
    strbuf=new char[STRINGSIZE];
}

OpenGL::~OpenGL()
{
    delete[] strbuf;
}

clString OpenGL::create_window(Roe*,
			       const clString& name,
			       const clString& width,
			       const clString& height)
{
    myname=name;
    width.get_int(xres);
    height.get_int(yres);
    return "opengl "+name+" -geometry "+width+"x"+height+" -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 2";
}

void OpenGL::redraw(Salmon* salmon, Roe* roe)
{
    // Start polygon counter...
    WallClockTimer timer;
    timer.clear();
    timer.start();

    // Get window information
    if(!tkwin){
	TCLTask::lock();
	tkwin=Tk_NameToWindow(the_interp, myname(), Tk_MainWindow(the_interp));
	if(!tkwin){
	    cerr << "Unable to locate window!\n";
	    TCLTask::unlock();
	    return;
	}
	dpy=Tk_Display(tkwin);
	win=Tk_WindowId(tkwin);
	cx=OpenGLGetContext(the_interp, myname());
	glXMakeCurrent(dpy, win, cx);
	current_drawer=this;
	int data[1];
	glGetIntegerv(GL_MAX_LIGHTS, data);
	maxlights=data[0];
	TCLTask::unlock();
    }

    // Make ourselves current
    if(current_drawer != this){
	current_drawer=this;
	TCLTask::lock();
	glXMakeCurrent(dpy, win, cx);
	TCLTask::unlock();
    }

    TCLTask::lock();

    // Clear the screen...
    Color bg(roe->bgcolor.get());
    glClearColor(bg.r(), bg.g(), bg.b(), 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup the view...
    View view(roe->view.get());
    double aspect=double(xres)/double(yres);
    double fovy=RtoD(2*Atan(aspect*Tan(DtoR(view.fov/2.))));

    DrawInfoOpenGL drawinfo;
    drawinfo.polycount=0;

    // Compute znear and zfar...
    double znear;
    double zfar;
    if(compute_depth(roe, view, znear, zfar)){
	glViewport(0, 0, xres, yres);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspect, znear, zfar);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Point eyep(view.eyep);
	Point lookat(view.lookat);
	Vector up(view.up);
	gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z());

	// Set up Lighting
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	Lighting& l=salmon->lighting;
	int idx=0;
	for(int i=0;i<l.lights.size();i++){
	    Light* light=l.lights[i];
	    light->opengl_setup(view, &drawinfo, idx);
	}
	for(i=0;i<idx && i<maxlights;i++)
	    glEnable(GL_LIGHT0+i);
	for(;i<GL_MAX_LIGHTS;i++)
	    glDisable(GL_LIGHT0+i);

	// Set up graphics state
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	clString shading(roe->shading.get());
//	GeomRenderMode::DrawType dt;
	if(shading == "wire"){
	    drawinfo.set_drawtype(DrawInfoOpenGL::WireFrame);
	    drawinfo.lighting=0;
	} else if(shading == "flat"){
	    drawinfo.set_drawtype(DrawInfoOpenGL::Flat);
	    drawinfo.lighting=0;
	} else if(shading == "gouraud"){
	    drawinfo.set_drawtype(DrawInfoOpenGL::Gouraud);
	    drawinfo.lighting=1;
	} else if(shading == "phong"){
	    drawinfo.set_drawtype(DrawInfoOpenGL::Phong);
	    drawinfo.lighting=1;
	} else {
	    cerr << "Unknown shading(" << shading << "), defaulting to gouraud" << endl;
	    drawinfo.set_drawtype(DrawInfoOpenGL::Gouraud);
	    drawinfo.lighting=1;
	}
	drawinfo.currently_lit=drawinfo.lighting;
	if(drawinfo.lighting)
	    glEnable(GL_LIGHTING);
	else
	    glDisable(GL_LIGHTING);
	drawinfo.pickmode=0;

	// Draw it all...
	drawinfo.push_matl(salmon->default_matl.get_rep());
	HashTableIter<int, PortInfo*> iter(&salmon->portHash);
	for (iter.first(); iter.ok(); ++iter) {
	    HashTable<int, SceneItem*>* serHash=iter.get_data()->objs;
	    HashTableIter<int, SceneItem*> serIter(serHash);
	    for (serIter.first(); serIter.ok(); ++serIter) {
		SceneItem *si=serIter.get_data();

		// Look up this object by name and see if it is supposed to be
		// displayed...
		ObjTag* vis;
		if(roe->visible.lookup(si->name, vis)){
		    if(vis->visible->get())
			si->obj->draw(&drawinfo);
		} 
 	    }
	}
	drawinfo.pop_matl();
    }

    // Show the pretty picture
    glXSwapBuffers(dpy, win);
    glXWaitGL();

    // Report statistics
    timer.stop();
    ostrstream str(strbuf, STRINGSIZE);
    str << "updatePerf " << roe->id << " \"";
    str << drawinfo.polycount << " polygons in " << timer.time()
	<< " seconds\" \"" << drawinfo.polycount/timer.time()
	<< " polygons/second\"" << '\0';
    TCL::execute(str.str());
    TCLTask::unlock();
}

void OpenGL::hide()
{
    tkwin=0;
    if(current_drawer==this)
	current_drawer=0;
}

void OpenGL::get_pick(Salmon* salmon, Roe* roe, int x, int y,
		      GeomObj*& pick_obj, GeomPick*& pick_pick)
{
    pick_obj=0;
    pick_pick=0;
    // Make ourselves current
    if(current_drawer != this){
	current_drawer=this;
	TCLTask::lock();
	glXMakeCurrent(dpy, win, cx);
	TCLTask::unlock();
    }
    // Setup the view...
    View view(roe->view.get());
    double aspect=double(xres)/double(yres);
    double fovy=RtoD(2*Atan(aspect*Tan(DtoR(view.fov/2.))));

    DrawInfoOpenGL drawinfo;
    drawinfo.polycount=0;

    // Compute znear and zfar...
    double znear;
    double zfar;
    if(compute_depth(roe, view, znear, zfar)){
	// Setup picking...
	TCLTask::lock();
	GLuint pick_buffer[pick_buffer_size];
	glSelectBuffer(pick_buffer_size, pick_buffer);
	glRenderMode(GL_SELECT);
	glInitNames();
	glPushName(0);

	glViewport(0, 0, xres, yres);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	int viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	gluPickMatrix(x, viewport[3]-y, pick_window, pick_window, viewport);
	gluPerspective(fovy, aspect, znear, zfar);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Point eyep(view.eyep);
	Point lookat(view.lookat);
	Vector up(view.up);
	gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z());

	drawinfo.lighting=0;
	drawinfo.set_drawtype(DrawInfoOpenGL::Flat);
	drawinfo.pickmode=1;

	// Draw it all...
	drawinfo.push_matl(salmon->default_matl.get_rep());
	HashTableIter<int, PortInfo*> iter(&salmon->portHash);
	for (iter.first(); iter.ok(); ++iter) {
	    HashTable<int, SceneItem*>* serHash=iter.get_data()->objs;
	    HashTableIter<int, SceneItem*> serIter(serHash);
	    for (serIter.first(); serIter.ok(); ++serIter) {
		SceneItem *si=serIter.get_data();

		// Look up this object by name and see if it is supposed to be
		// displayed...
		if(si->obj->get_pick()){
		    glLoadName((GLuint)si->obj);
		    si->obj->draw(&drawinfo);
		}
	    }
	}
	drawinfo.pop_matl();
	glFlush();
	int hits=glRenderMode(GL_RENDER);
	TCLTask::unlock();
	GLuint min_z;
	GLuint hit_obj=0;
	GLuint hit_pick=0;
	if(hits >= 1){
	    int idx=0;
	    min_z=pick_buffer[1];
	    hit_obj=pick_buffer[3];
	    hit_pick=pick_buffer[4];
	    for (int h=0; h<hits; h++) {
		int nnames=pick_buffer[idx++];
		ASSERT(nnames >= 2);
		GLuint z=pick_buffer[idx++];
		if (h==0 || z < min_z) {
		    min_z=z;
		    idx++; // Skip Max Z
		    hit_obj=pick_buffer[idx++];
		    idx+=nnames-2; // Skip to the last one...
		    hit_pick=pick_buffer[idx++];
		} else {
		    idx+=nnames+1;
		}
	    }
	}
	pick_obj=(GeomObj*)hit_obj;
	pick_pick=(GeomPick*)hit_pick;
    }
}

void OpenGL::put_scanline(int y, int width, Color* scanline, int repeat)
{
    float* pixels=new float[width*3];
    float* p=pixels;
    for(int i=0;i<width;i++){
	*p++=scanline[i].r();
	*p++=scanline[i].g();
	*p++=scanline[i].b();
    }
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslated(-1, -1, 0);
    glScaled(2./xres, 2./yres, 1.0);
    glDepthFunc(GL_ALWAYS);
    glDrawBuffer(GL_FRONT);
    for(i=0;i<repeat;i++){
	glRasterPos2i(0, y+i);
	glDrawPixels(width, 1, GL_RGB, GL_FLOAT, pixels);
    }
    glDepthFunc(GL_LEQUAL);
    glDrawBuffer(GL_BACK);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    delete[] pixels;
}
