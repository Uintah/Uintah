
/*
 *  Postscript.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <PSECommon/Modules/Salmon/Renderer.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECommon {
namespace Modules {

using SCICore::Containers::clString;

class Postscript : public Renderer {
public:
    Postscript();
    virtual ~Postscript();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void old_redraw(Salmon*, Roe*);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&, int&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat);
    virtual void hide();
};

static Renderer* make_Postscript()
{
    return scinew Postscript;
}

static int query_Postscript()
{
    return 1;
}

RegisterRenderer Postscript_renderer("Postscript", &query_Postscript,
				     &make_Postscript);

Postscript::Postscript()
{
}

Postscript::~Postscript()
{
}

clString Postscript::create_window(Roe*,
				   const clString& name,
				   const clString& width,
				   const clString& height)
{
    return "canvas "+name+" -width "+width+" -height "+height+" -background lavender";
}

void Postscript::old_redraw(Salmon*, Roe*)
{
    NOT_FINISHED("Postscript::redraw");
}

void Postscript::hide()
{
}

void Postscript::get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&, int&)
{
    NOT_FINISHED("Postscript::get_pick");
}

void Postscript::put_scanline(int, int, Color*, int)
{
    NOT_FINISHED("Postscript::put_scanline");
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:37  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:51  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
