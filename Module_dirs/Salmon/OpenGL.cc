
#include <Modules/Salmon/Renderer.h>
#include <Classlib/NotFinished.h>

class OpenGL : public Renderer {
public:
    OpenGL();
    virtual ~OpenGL();
    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
};

static Renderer* make_OpenGL()
{
    return new OpenGL;
}

RegisterRenderer OpenGL_renderer("OpenGL", &make_OpenGL);

OpenGL::OpenGL()
{
}

OpenGL::~OpenGL()
{
}

clString OpenGL::create_window(const clString& name,
				   const clString& width,
				   const clString& height)
{
    return "opengl "+name+" -geometry "+width+"x"+height+" -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 2";
}

void OpenGL::redraw(Salmon*, Roe*)
{
    NOT_FINISHED("OpenGL::redraw");
}
