
#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Salmon.h>
#include <Modules/Salmon/Roe.h>
#include <Classlib/Array1.h>
#include <Classlib/AVLTree.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Geometry/Transform.h>
#include <Geom/Geom.h>
#include <Geom/GeomX11.h>
#include <Math/Trig.h>
#include <TCL/TCLTask.h>

#include <X11/Xlib.h>
#include <tcl/tcl7.3/tcl.h>
#include <tcl/tk3.6/tk.h>
#include <stdlib.h>
#include <strstream.h>

const int STRINGSIZE=100;

extern Tcl_Interp* the_interp;

class X11 : public Renderer {
    clString windowname;
    Tk_Window tkwin;
    DrawInfoX11* drawinfo;
    int ncolors;
    XColor** tkcolors;
    char* strbuf;

    void setup_window();
public:
    X11();
    virtual ~X11();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon* salmon, Roe* roe);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat);
    virtual void hide();
};

static Renderer* make_X11()
{
    return new X11;
}

static int query_X11()
{
    return 1;
}

RegisterRenderer X11_renderer("X11", &query_X11, &make_X11);

X11::X11()
: tkwin(0), drawinfo(0), tkcolors(0)
{
    strbuf=new char[STRINGSIZE];
}

X11::~X11()
{
    delete[] strbuf;
    if(drawinfo){
	if(drawinfo->colors)
	    delete[] drawinfo->colors;
	delete drawinfo;
    }
}

clString X11::create_window(Roe*,
			    const clString& name,
			    const clString& width,
			    const clString& height)
{
    windowname=name;
    width.get_int(xres);
    height.get_int(yres);
    return "canvas "+name+" -width "+width+" -height "+height+" -background black";
}

void X11::setup_window()
{
    TCLTask::lock();
    tkwin=Tk_NameToWindow(the_interp, windowname(),
			  Tk_MainWindow(the_interp));
    // Set up drawinfo....
    if(tkcolors){
	for(int i=0;i<ncolors;i++){
	    Tk_FreeColor(tkcolors[i]);
	}
    }
    if(!drawinfo)
	drawinfo=new DrawInfoX11;
    else if(drawinfo->colors)
	delete[] drawinfo->colors;
    drawinfo->dpy=Tk_Display(tkwin);
    drawinfo->win=Tk_WindowId(tkwin);
    drawinfo->gc=XCreateGC(drawinfo->dpy, drawinfo->win, 0, 0);
    if(!drawinfo->gc){
	TCLTask::unlock();
	cerr << "Error allocating GC!\nn";
	return;
    }


    int sr=8;
    int sg=8;
    int sb=8;
    while(1){
	int failed=0;
	ncolors=sr*sg*sb;
	tkcolors=new XColor*[ncolors];
	int idx=0;
	cerr << sr << "x" << sg << "x" << sb << endl;
	for(int i=0;i<sr && !failed;i++){
	    for(int j=0;j<sg && !failed;j++){
		for(int k=0;k<sb && !failed;k++){
		    XColor pref;
		    pref.red=i*65535/(sr-1);
		    pref.green=j*65535/(sg-1);
		    pref.blue=k*65535/(sb-1);
		    pref.flags = DoRed|DoGreen|DoBlue;
		    XColor* c=Tk_GetColorByValue2(the_interp, tkwin,
						  Tk_Colormap(tkwin), &pref);
		    if(!c){
			failed=1;
		    } else {
			tkcolors[idx++]=c;
		    }
		}
	    }
	}
	if(failed){
	    for (int i=0;i<idx;i++)
		Tk_FreeColor(tkcolors[i]);
	    delete[] tkcolors;
	    if(sr == sg && sr == sb){
		sb--;
	    } else if(sr == sg){
		sg--;
	    } else {
		sr--;
	    }
	    if(sb == 0){
		tkcolors=0;
		cerr << "Cannot allocate enough colors to survive!\n";
		exit(-1);
	    }
	} else {
	    break;
	}
    }
    drawinfo->red_max=sr-1;
    drawinfo->green_max=sg-1;
    drawinfo->blue_max=sb-1;
    drawinfo->red_mult=sg*sb;
    drawinfo->green_mult=sb;
    drawinfo->colors=new unsigned long[ncolors];
    for(int i=0;i<ncolors;i++)
	drawinfo->colors[i]=tkcolors[i]->pixel;
    TCLTask::unlock();
}

void X11::redraw(Salmon* salmon, Roe* roe)
{
    WallClockTimer timer;
    timer.start();
    if(!tkwin){
	setup_window();
    }
    Array1<GeomObj*> free;
    Array1<GeomObj*> dontfree;
    HashTableIter<int, PortInfo*> portiter(&salmon->portHash);
    for(portiter.first();portiter.ok();++portiter){
	HashTableIter<int, SceneItem*> objiter(portiter.get_data()->objs);
	for(objiter.first();objiter.ok();++objiter){
	    SceneItem* si=objiter.get_data();
	    si->obj->make_prims(free, dontfree);
	}
    }
    int npolys=free.size()+dontfree.size();

    View view(roe->view.get());

    // Compute znear and zfar...
    double znear;
    double zfar;
    if(compute_depth(roe, view, znear, zfar)){
	cerr << "znear=" << znear << ", zfar=" << zfar << endl;
	Transform trans;
	trans.load_identity();
	trans.perspective(view.eyep, view.lookat,
			  view.up, view.fov,
			  znear, zfar, xres, yres);
	drawinfo->transform=&trans;
	drawinfo->current_matl=salmon->default_matl.get_rep();
	drawinfo->current_lit=1;
	drawinfo->view=view;
	NOT_FINISHED("Light source selection");
	drawinfo->lighting=salmon->lighting;
	AVLTree<double, GeomObj*> objs;
	double zmin=1000;
	double zmax=-1000;
	for(int i=0;i<free.size();i++){
	    GeomObj* obj=free[i];
	    objs.insert(-obj->depth(drawinfo), obj);
	    zmin=Min(zmin, -obj->depth(drawinfo));
	    zmax=Max(zmax, -obj->depth(drawinfo));
	}
	for(i=0;i<dontfree.size();i++){
	    GeomObj* obj=dontfree[i];
	    objs.insert(-obj->depth(drawinfo), obj);
	    zmin=Min(zmin, -obj->depth(drawinfo));
	    zmax=Max(zmax, -obj->depth(drawinfo));
	}
	cerr << "Z goes from " << zmin << " to " << zmax << endl;

	AVLTreeIter<double, GeomObj*> iter(&objs);
	TCLTask::lock();
	XClearWindow(drawinfo->dpy, drawinfo->win);
	for(iter.first();iter.ok();++iter){
	    GeomObj* obj=iter.get_data();
	    obj->draw(drawinfo);
	}
	TCLTask::unlock();
	for(i=0;i<free.size();i++){
	    delete free[i];
	}
    } else {
	TCLTask::lock();
	XClearWindow(drawinfo->dpy, drawinfo->win);
	TCLTask::unlock();
    }
    timer.stop();
    ostrstream str(strbuf, STRINGSIZE);
    str << "updatePerf " << roe->id << " \"";
    str << npolys << " polygons in " << timer.time()
	<< " seconds\" \"" << npolys/timer.time()
	<< " polygons/second\"" << '\0';
    cerr << "str=" << str.str() << endl;
    TCL::execute(str.str());
}

void X11::hide()
{
    tkwin=0;
}

void X11::get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&)
{
    NOT_FINISHED("X11::get_pick");
}

void X11::put_scanline(int, int, Color*, int)
{
    NOT_FINISHED("X11::put_scanline");
}
