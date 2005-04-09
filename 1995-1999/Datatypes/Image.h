

#ifndef SCI_DATATYPES_IMAGE_H
#define SCI_DATATYPES_IMAGE_H 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array2.h>
#include <Geom/Color.h>

class Image;
typedef LockingHandle<Image> ImageHandle;

class Image : public Datatype {
    /* Complex... */
public:
    float** rows;
    int xr, yr;
    Image(int xres, int yres);
    Image(const Image&);
    virtual ~Image();
    int xres() const;
    int yres() const;

    inline float getr(int x, int y) {
	return rows[y][x*2];
    }
    inline void set(int x, int y, float r, float i) {
	rows[y][x*2]=r;
	rows[y][x*2+1]=i;
    }
    float max_abs();

    virtual Image* clone();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class ColorImage {
public:
    ColorImage(int xres, int yres);
    ~ColorImage();
    Array2<Color> imagedata;
    inline Color& get_pixel(int x, int y) {
	return imagedata(y,x);
    }
    inline void put_pixel(int x, int y, const Color& pixel) {
	imagedata(y,x)=pixel;
    }
    int xres() const;
    int yres() const;
};

class DepthImage {
public:
    DepthImage(int xres, int yres);
    ~DepthImage();
    Array2<double> depthdata;
    double get_depth(int x, int y) {
	return depthdata(y,x);
    }
    inline void put_pixel(int x, int y, double depth) {
	depthdata(y,x)=depth;
    }
    int xres() const;
    int yres() const;
};

#endif
