
#include <Salmon/Renderer.h>
#include <NotFinished.h>

class Postscript : public Renderer {
public:
    Postscript();
    virtual ~Postscript();
    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
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
    NOT_FINISHED("X11::redraw");
}
