
#include <Modules/Salmon/Renderer.h>
#include <Classlib/NotFinished.h>

class Postscript : public Renderer {
public:
    Postscript();
    virtual ~Postscript();
    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&);
    virtual void hide();
};

static Renderer* make_Postscript()
{
    return new Postscript;
}

RegisterRenderer Postscript_renderer("Postscript", &make_Postscript);

Postscript::Postscript()
{
}

Postscript::~Postscript()
{
}

clString Postscript::create_window(const clString& name,
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
