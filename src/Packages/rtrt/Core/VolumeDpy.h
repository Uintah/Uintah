
#ifndef VOLUMEDPY_H
#define VOLUMEDPY_H 1

#include <Core/Thread/Runnable.h>
#include <X11/Xlib.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

using SCIRun::Runnable;

class VolumeBase;

class VolumeDpy : public Runnable {
    Array1<VolumeBase*> vols;
    int* hist;
    int histmax;
    int xres, yres;
    float datamin, datamax;
    void compute_hist(unsigned int fid);
    void draw_hist(unsigned int fid, XFontStruct* font_struct,
		   bool redraw_hist);
    void move(int x, int y);
    float new_isoval;
public:
    float isoval;
    VolumeDpy(float isoval=-123456);
    void attach(VolumeBase*);
    virtual ~VolumeDpy();
    virtual void run();
    void animate(bool& changed);
};

} // end namespace rtrt


#endif
