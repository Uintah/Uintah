
#include <Modules/HP/ScalarFieldHP.h>
#include <Classlib/NotFinished.h>
#include <tiffio.h>
#include <iostream.h>
#include <values.h>

ScalarFieldHP::ScalarFieldHP()
    : ScalarField(ScalarField::HP)
{
    width=height=-1;
}

ScalarFieldHP::ScalarFieldHP(const ScalarFieldHP& copy)
    : ScalarField(ScalarField::HP)
{
    NOT_FINISHED("ScalarFieldHP::ScalarFieldHP");
}

ScalarFieldHP::~ScalarFieldHP()
{
}

ScalarField* ScalarFieldHP::clone()
{
    return new ScalarFieldHP(*this);
}

void ScalarFieldHP::read_image(const clString& file)
{
    TIFF *tif = TIFFOpen(file(), "r");
    if (!tif) {
	ASSERT(!"File not found or not a TIFF file..");
    }

    int xdim,ydim;
    uint16 bps,spp;

    unsigned long imagelength;
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &xdim);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &ydim);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    if(width==-1){
	width=TIFFScanlineSize(tif);
	height=(int)imagelength;
    } else {
	ASSERTEQ(width, TIFFScanlineSize(tif));
	ASSERTEQ(height, imagelength);
    }
	
    cerr << "--TiffReader--\n";
    cerr << "Reading TIFF file : " << file << "\n";
    cerr << "Dimensions: " << xdim << " " << ydim << "\n";
    cerr << "spp: " << spp << "  bps: " << bps << "\n";

    ASSERTEQ(bps, 8);
    ASSERTEQ(spp, 1);

    unsigned char* pic = new unsigned char[TIFFScanlineSize(tif)*imagelength];
    images.add(pic);
    cerr << "scanlinesize : " << TIFFScanlineSize(tif) << "\n";
    unsigned char* buf=pic;
    for (int row = 0; row < imagelength; row++){
	TIFFReadScanline(tif, buf, row, 0);
	buf+=TIFFScanlineSize(tif);
    }
    TIFFClose(tif);
}

void ScalarFieldHP::compute_minmax()
{
    data_min=MAXDOUBLE;
    data_max=-MAXDOUBLE;
    int n=width*height;
    cerr << "There are " << images.size() << " images\n";
    for(int i=0;i<images.size();i++){
	unsigned char* p=images[i];
	for(int j=0;j<n;j++){
	    if(p[j] < data_min)
		data_min=p[j];
	    if(p[j] > data_max)
		data_max=p[j];
	}
    }
}

void ScalarFieldHP::compute_bounds()
{
    bmin=Point(-1,-1,0);
    bmax=Point(1,1,1);
}

int ScalarFieldHP::interpolate(const Point& p, double& value, int&,
			       double, double, int)
{
    double z=(int)(p.z()*height);
    int iz=(int)z;
    if(p.z() < 0 || p.z() > 1)
	return 0;
    double r=sqrt(p.x()*p.x()+p.y()*p.y());
    if(r > 1)
	return 0;
    double theta=atan2(p.y(), p.x());
    if(theta < 0){
	theta+=M_PI;
	r=(1.-r)/2.;
    } else {
	r=(r+1.)/2.;
    }
    double ix=(int)(r*(width-1));
    double slice=theta/M_PI*(images.size()-2);
    int islice=(int)slice;
    unsigned char* buf=images[islice];
    int idx=(iz*width+ix);
    value=buf[idx];
    return 1;
}

int ScalarFieldHP::interpolate(const Point& p, double& value,
			       double eps1, double eps2)
{
    int ix=0;
    return interpolate(p, value, ix, eps1, eps2);
}

void ScalarFieldHP::io(Piostream&)
{
    NOT_FINISHED("ScalarFieldHP::io");
}

Vector ScalarFieldHP::gradient(const Point& p)
{
    NOT_FINISHED("ScalarFieldHP::gradient");
}

void ScalarFieldHP::get_boundary_lines(Array1<Point>& lines)
{
    NOT_FINISHED("ScalarFieldHP::get_boundary_lines");
}
