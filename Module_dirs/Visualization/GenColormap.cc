
/*
 *  GenColormap.cc:  Generate Color maps
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Module.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/Colormap.h>
#include <TCL/TCLvar.h>

class GenColormap : public Module {
    ColormapOPort* outport;

    TCLstring map_type;
    TCLint nlevels;

    // General Parameters
    TCLdouble ambient_intens;
    TCLdouble diffuse_intens;
    TCLdouble specular_intens;
    TCLdouble shininess;
    TCLdouble spec_percent;
    TCLColor spec_color;

    // Parameters for rainbow colormap type
    TCLdouble rainbow_hue_min;
    TCLdouble rainbow_hue_max;
    TCLdouble rainbow_sat;
    TCLdouble rainbow_val;

    void compute_colors(const ColormapHandle& cmap, int idx,
			const Color& color);
public:
    GenColormap(const clString& id);
    GenColormap(const GenColormap&, int deep);
    virtual ~GenColormap();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_GenColormap(const clString& id)
{
    return new GenColormap(id);
}

static RegisterModule db1("Visualization", "GenColormap", make_GenColormap);

GenColormap::GenColormap(const clString& id)
: Module("GenColormap", id, Source), map_type("map_type", id, this),
  nlevels("nlevels", id, this), ambient_intens("ambient_intens", id, this),
  diffuse_intens("diffuse_intens", id, this),
  specular_intens("specular_intens", id, this),
  shininess("shininess", id, this),
  spec_percent("spec_percent", id, this), spec_color("spec_color", id, this),
  rainbow_hue_min("rainbow_hue_min", id, this),
  rainbow_hue_max("rainbow_hue_max", id, this),
  rainbow_sat("rainbow_sat", id, this), rainbow_val("rainbow_val", id, this)
{
    // Create the output port
    outport=new ColormapOPort(this, "Colormap", ColormapIPort::Atomic);
    add_oport(outport);
}

GenColormap::GenColormap(const GenColormap& copy, int deep)
: Module(copy, deep), map_type("map_type", id, this),
  nlevels("nlevels", id, this), ambient_intens("ambient_intens", id, this),
  diffuse_intens("diffuse_intens", id, this),
  specular_intens("specular_intens", id, this),
  shininess("shininess", id, this),
  spec_percent("spec_percent", id, this), spec_color("spec_color", id, this),
  rainbow_hue_min("rainbow_hue_min", id, this),
  rainbow_hue_max("rainbow_hue_max", id, this),
  rainbow_sat("rainbow_sat", id, this), rainbow_val("rainbow_val", id, this)
{
    NOT_FINISHED("GenColormap::GenColormap");
}

GenColormap::~GenColormap()
{
}

Module* GenColormap::clone(int deep)
{
    return new GenColormap(*this, deep);
}

void GenColormap::execute()
{
    int nl=nlevels.get();
    double range_min=0;
    double range_max=1;
    NOT_FINISHED("Colormap ranges");
    ColormapHandle cmap(new Colormap(nl, range_min, range_max));
    clString mt(map_type.get());
    if(mt=="rainbow" || mt==""){
	// Compute a colormap which varies the hue in HSV color space
	double hue_min=rainbow_hue_min.get();
	double hue_max=rainbow_hue_max.get();
	double hue_range=hue_max-hue_min;
	double sat=rainbow_sat.get();
	double val=rainbow_val.get();
	for(int i=0;i<nl;i++){
	    double hue=double(i)/double(nl-1)*hue_range+hue_min;
	    compute_colors(cmap, i, HSVColor(hue, sat, val));
	}
    } else if(mt=="voltage"){
	// Compute a color ramp which goes from blue to green to red
	for(int i=0;i<nl;i++){
	    double p=double(i)/double(nl-1);
	    double red=0;
	    double green=0;
	    double blue=0;
	    if(p<0.5){
		blue=1-2*p;
		green=2*p;
	    } else {
		red=2*p-1;
		green=2-2*p;
	    }
	    compute_colors(cmap, i, Color(red, green, blue));
	}
    } else {
	error("Unknown colormap type!");
    }
    outport->send(cmap);
}

void GenColormap::compute_colors(const ColormapHandle& cmap, int idx,
				 const Color& color)
{
    Color ambient(color*ambient_intens.get());
    Color diffuse(color*diffuse_intens.get());
    Color spec1(color*specular_intens.get());
    Color spec2(spec_color.get());
    double sp=spec_percent.get();
    Color specular(spec1*sp+spec2*(1-sp));
    cmap->colors[idx]=new Material(ambient, diffuse, specular, shininess.get());
}
