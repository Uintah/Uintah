
#include <Salmon/Renderer.h>
#include <Salmon/Salmon.h>
#include <Salmon/Roe.h>
#include <NotFinished.h>
#include <X11/Xlib.h>
#include <TCLTask.h>
#include <tcl.h>
#include <tk.h>
#include <Classlib/Array1.h>
#include <Classlib/AVLTree.h>
#include <Classlib/HashTable.h>
#include <Geometry/Transform.h>
#include <Geom.h>

extern Tcl_Interp* the_interp;

class X11 : public Renderer {
    int have_win;
    clString windowname;
    Tk_Window tkwin;
    Display* dpy;
    Window win;
    GC gc;
public:
    X11();
    virtual ~X11();
    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon* salmon, Roe* roe);
};

static Renderer* make_X11()
{
    return new X11;
}

RegisterRenderer X11_renderer("X11", &make_X11);

X11::X11()
{
}

X11::~X11()
{
}

clString X11::create_window(const clString& name,
			    const clString& width,
			    const clString& height)
{
    have_win=0;
    windowname=name;
    return "canvas "+name+" -width "+width+" -height "+height+" -background black";
}

void X11::redraw(Salmon* salmon, Roe* roe)
{
    WallClockTimer timer;
    timer.start();
    if(!have_win){
	TCLTask::lock();
	have_win=1;
	tkwin=Tk_NameToWindow(the_interp, windowname(),
			      Tk_MainWindow(the_interp));
	dpy=Tk_Display(tkwin);
	win=Tk_WindowId(tkwin);
	gc=XCreateGC(dpy, win, 0, 0);
	if(!gc){
	    TCLTask::unlock();
	    cerr << "Error allocating GC!\nn";
	    return;
	}
	TCLTask::unlock();
    }
    Array1<GeomObj*> free;
    Array1<GeomObj*> dontfree;
    HashTableIter<int, HashTable<int, GeomObj*>*> portiter(&salmon->portHash);
    for(portiter.first();portiter.ok();++portiter){
	HashTableIter<int, GeomObj*> objiter(portiter.get_data());
	for(objiter.first();objiter.ok();++objiter){
	    GeomObj* obj=objiter.get_data();
	    obj->make_prims(free, dontfree);
	}
    }
    int npolys=free.size()+dontfree.size();
    // Set up drawinfo....
    DrawInfo drawinfo;
    drawinfo.dpy=dpy;
    drawinfo.win=win;
    drawinfo.gc=gc;
    double t[16];
    t[0]=4000; t[1]=0;   t[2]=0;   t[3]=-1900;
    t[4]=0;   t[5]=4000; t[6]=0;   t[7]=-1750;
    t[8]=0;   t[9]=0;   t[10]=4000;t[11]=-2000;
    t[12]=0;  t[13]=0;  t[14]=0;  t[15]=1.0;
    Transform transform;
    transform.set(t);
    drawinfo.transform=&transform;
    AVLTree<double, GeomObj*> objs;
    for(int i=0;i<free.size();i++){
	GeomObj* obj=free[i];
	objs.insert(obj->depth(&drawinfo), obj);
    }
    for(i=0;i<dontfree.size();i++){
	GeomObj* obj=dontfree[i];
	objs.insert(obj->depth(&drawinfo), obj);
    }

    AVLTreeIter<double, GeomObj*> iter(&objs);
    TCLTask::lock();
    XClearWindow(dpy, win);
    for(iter.first();iter.ok();++iter){
	GeomObj* obj=iter.get_data();
	obj->draw_X11(&drawinfo);
    }
    TCLTask::unlock();
    for(i=0;i<free.size();i++){
	delete free[i];
    }
    timer.stop();
    clString perf1(to_string(npolys)+" polygons in "+to_string(timer.time())+" seconds");
    double pps=double(npolys)/timer.time();
    clString perf2(to_string(pps)+" polys/sec");
    static clString q("\"");
    static clString s(" ");
    static clString c("updatePerf ");
    TCL::execute(c+roe->id+s+q+perf1+q+s+q+perf2+q);
}
