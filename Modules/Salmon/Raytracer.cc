
#include <Modules/Salmon/Renderer.h>
#include <Classlib/NotFinished.h>

class Raytracer : public Renderer {
public:
    Raytracer();
    virtual ~Raytracer();
    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
    virtual void hide();
};

static Renderer* make_Raytracer()
{
    return new Raytracer;
}

RegisterRenderer Raytracer_renderer("Raytracer", &make_Raytracer);

Raytracer::Raytracer()
{
}

Raytracer::~Raytracer()
{
}

clString Raytracer::create_window(const clString& name,
				   const clString& width,
				   const clString& height)
{
    return "canvas "+name+" -width "+width+" -height "+height+" -background RoyalBlue";
}

void Raytracer::redraw(Salmon*, Roe*)
{
    NOT_FINISHED("X11::redraw");
}

void Raytracer::hide()
{
}
