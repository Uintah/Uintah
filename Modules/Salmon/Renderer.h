/*
 *  Renderer.h: Abstract interface to a renderer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Renderer_h
#define SCI_project_Renderer_h

#include <Classlib/AVLTree.h>
#include <Classlib/String.h>
class Color;
class GeomObj;
class GeomPick;
class Renderer;
class Roe;
class Salmon;
class View;

typedef Renderer* (*make_Renderer)();
typedef int (*query_Renderer)();
class RegisterRenderer;
class TCLArgs;

class Renderer {
public:
    static Renderer* create(const clString& type);
    static AVLTree<clString, RegisterRenderer*>* get_db();

    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height)=0;
    virtual void redraw(Salmon*, Roe*);
    virtual void redraw(Salmon*, Roe*, double tbeg, double tend,
			int nframes, double framerate);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&)=0;
    virtual void hide()=0;
    virtual void dump_image(const clString&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat=1)=0;
    virtual void listvisuals(TCLArgs&);
    virtual void setvisual(const clString& wname, int i, int width, int height);

    int compute_depth(Roe* roe, const View& view, double& near, double& far);

    int xres, yres;
};

class RegisterRenderer {
public:
    clString name;
    query_Renderer query;
    make_Renderer maker;
    RegisterRenderer(const clString& name, query_Renderer tester,
		     make_Renderer maker);
    ~RegisterRenderer();
};

#endif
