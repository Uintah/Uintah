
#include <Modules/Salmon/Renderer.h>
#include <Classlib/NotFinished.h>

class Postscript : public Renderer {
public:
    Postscript();
    virtual ~Postscript();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat);
    virtual void hide();
};

static Renderer* make_Postscript()
{
    return new Postscript;
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

void Postscript::redraw(Salmon*, Roe*)
{
    NOT_FINISHED("Postscript::redraw");
}

void Postscript::hide()
{
}

void Postscript::get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&)
{
    NOT_FINISHED("Postscript::get_pick");
}

void Postscript::put_scanline(int, int, Color*, int)
{
    NOT_FINISHED("Postscript::put_scanline");
}
