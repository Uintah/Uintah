
/*
 *  GenColorMap.cc:  Generate Color maps
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Apr. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Dataflow/Module.h>
#include <Classlib/NotFinished.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/ColorMap.h>
#include <Geom/Material.h>
#include <Geom/TCLGeom.h>
#include <Malloc/Allocator.h>
#include <Math/CatmullRomSpline.h>

#include <TCL/TCLvar.h>

class MaterialKey : public Material {
public:
   int hsv;
   double ratio;
   
   MaterialKey();
   MaterialKey( const double r );
   MaterialKey( const MaterialKey& );
   MaterialKey( const Material& );
   ~MaterialKey();
   MaterialKey operator*( const double ) const;
   MaterialKey operator+( const MaterialKey& ) const;
   MaterialKey operator-( const MaterialKey& ) const;
   MaterialKey& operator=( const MaterialKey& );
   MaterialKey& operator=( const Material& );
};


class GenColorMap : public Module {
   ColorMapOPort* outport;
   
   Array1<MaterialKey> keys;
   CatmullRomSpline<MaterialKey> spline;
   
   TCLint nlevels;
   
   TCLstring interp_type;
   TCLstring cinterp_type;
   
   TCLMaterial tclmat;

   TCLdouble min;
   TCLdouble max;

   TCLint gen_rangemap;
   
   Material find_level( const double );
   void gen_map( const clString& mt );
   int making_map;
   
public:
   GenColorMap( const clString& id );
   GenColorMap( const GenColorMap&, int deep );
   virtual ~GenColorMap();
   virtual Module* clone( int deep );
   virtual void execute();

   void init_tcl();
   void tcl_command( TCLArgs&, void* );
};

MaterialKey::MaterialKey()
: Material(), hsv(0), ratio(0)
{
}

MaterialKey::MaterialKey( const double r )
: Material(), hsv(0), ratio(r)
{
}

MaterialKey::MaterialKey( const MaterialKey& copy )
: Material(copy), hsv(copy.hsv), ratio(copy.ratio)
{
}

MaterialKey::MaterialKey( const Material& copy )
: Material(copy), hsv(0), ratio(0)
{
}

MaterialKey::~MaterialKey()
{
}

MaterialKey
MaterialKey::operator*( const double x ) const
{
   MaterialKey mk;

   if (hsv) {
      mk.ambient = (HSVColor)ambient*x;
      mk.diffuse = (HSVColor)diffuse*x;
      mk.specular = (HSVColor)specular*x;
      mk.emission = (HSVColor)emission*x;
      mk.hsv=1;
   } else {
      mk.ambient = ambient*x;
      mk.diffuse = diffuse*x;
      mk.specular = specular*x;
      mk.emission = emission*x;
   }
   mk.shininess = shininess*x;
   mk.reflectivity = reflectivity*x;
   mk.transparency = transparency*x;
   mk.refraction_index = refraction_index*x;
   
   return mk;
}

MaterialKey
MaterialKey::operator+( const MaterialKey& k ) const
{
   MaterialKey mk;

   if (hsv) {
      mk.ambient = (HSVColor)ambient+(HSVColor)k.ambient;
      mk.diffuse = (HSVColor)diffuse+(HSVColor)k.diffuse;
      mk.specular = (HSVColor)specular+(HSVColor)k.specular;
      mk.emission = (HSVColor)emission+(HSVColor)k.emission;
      mk.hsv=1;
   } else {
      mk.ambient = ambient+k.ambient;
      mk.diffuse = diffuse+k.diffuse;
      mk.specular = specular+k.specular;
      mk.emission = emission+k.emission;
   }
   mk.shininess = shininess+k.shininess;
   mk.reflectivity = reflectivity+k.reflectivity;
   mk.transparency = transparency+k.transparency;
   mk.refraction_index = refraction_index+k.refraction_index;

   return mk;
}

MaterialKey
MaterialKey::operator-( const MaterialKey& k ) const
{
   return *this + k*-1;
}

MaterialKey&
MaterialKey::operator=( const MaterialKey& m )
{
   Material::operator=(m);
   hsv = m.hsv;
   ratio = m.ratio;
   return *this;
}

MaterialKey&
MaterialKey::operator=( const Material& m )
{
   Material::operator=(m);
   return *this;
}

extern "C" {
Module* make_GenColorMap( const clString& id )
{
    return scinew GenColorMap(id);
}
}

GenColorMap::GenColorMap( const clString& id )
: Module("GenColorMap", id, Source),
  nlevels("nlevels", id, this),  tclmat("material", id, this),
  interp_type("interp_type", id, this), cinterp_type("cinterp_type", id, this),
  min("min", id, this), max("max", id, this),
  gen_rangemap("gen_rangemap", id, this),
  keys(2), making_map(0)
{
   keys[0].ratio = 0;
   keys[1].ratio = 1;

   spline.setData(keys);
   
   // Create the output port
   outport=scinew ColorMapOPort(this, "ColorMap", ColorMapIPort::Atomic);
   add_oport(outport);

}

GenColorMap::GenColorMap( const GenColorMap& copy, int deep )
: Module(copy, deep),
  nlevels("nlevels", id, this), tclmat("material", id, this),
  interp_type("interp_type", id, this), cinterp_type("cinterp_type", id, this),
  min("min", id, this), max("max", id, this), gen_rangemap("gen_rangemap", id, this),
  keys(copy.keys), spline(copy.spline), making_map(0)
{
   NOT_FINISHED("GenColorMap::GenColorMap");
}

GenColorMap::~GenColorMap()
{
}

Module*
GenColorMap::clone( int deep )
{
    return scinew GenColorMap(*this, deep);
}

void
GenColorMap::tcl_command( TCLArgs& args, void* userdata)
{
   int i;
   double f;
   
   if (args.count() < 2) {
      args.error("GenColorMap needs a minor command");
      return;
   } else if (args[1] == "getmat") {
      if (args.count() != 3) {
	 args.error("GenColorMap needs index");
	 return;
      }
      if (!args[2].get_int(i)) {
	 args.error("GenColorMap can't parse index `"+args[2]+"'");
	 return;
      }
      if ((i < 0) || (i >= keys.size())) {
	 args.error("GenColorMap index out of range `"+args[2]+"'");
	 return;
      }
      tclmat.set(keys[i]);
   } else if (args[1] == "setmat") {
      if (args.count() != 3) {
	 args.error("GenColorMap needs index");
	 return;
      }
      if (!args[2].get_int(i)) {
	 args.error("GenColorMap can't parse index `"+args[2]+"'");
	 return;
      }
      if ((i < 0) || (i >= keys.size())) {
	 args.error("GenColorMap index out of range `"+args[2]+"'");
	 return;
      }
      reset_vars();
      keys[i] = tclmat.get();
      spline[i] = keys[i];
   } else if (args[1] == "getslice") {
      if (args.count() != 3) {
	 args.error("GenColorMap needs ratio");
	 return;
      }
      if (!args[2].get_int(i)) {
	 args.error("GenColorMap can't parse ratio `"+args[2]+"'");
	 return;
      }
      if ((i < 0) || (i >= nlevels.get())) {
	 args.error("GenColorMap ratio out of range `"+args[2]+"'");
	 return;
      }
      Array1<clString> result(14);
      double f(i/(double)(nlevels.get()-1));
      Material m(find_level(f));
      result[0] = to_string(int(65535*m.ambient.r()));
      result[1] = to_string(int(65535*m.ambient.g()));
      result[2] = to_string(int(65535*m.ambient.b()));
      result[3] = to_string(int(65535*m.diffuse.r()));
      result[4] = to_string(int(65535*m.diffuse.g()));
      result[5] = to_string(int(65535*m.diffuse.b()));
      result[6] = to_string(int(65535*m.specular.r()));
      result[7] = to_string(int(65535*m.specular.g()));
      result[8] = to_string(int(65535*m.specular.b()));
      result[9] = to_string(int(65535*m.emission.r()));
      result[10] = to_string(int(65535*m.emission.g()));
      result[11] = to_string(int(65535*m.emission.b()));
      result[12] = to_string(int(65535*m.reflectivity));
      result[13] = to_string(int(65535*m.transparency));

      args.result(args.make_list(result));
   } else if (args[1] == "nlevels") {
      if (args.count() != 3) {
	 args.error("GenColorMap needs nlevels");
	 return;
      }
      if (!args[2].get_int(i)) {
	 args.error("GenColorMap can't parse nlevels `"+args[2]+"'");
	 return;
      }
      if ((i < 0) || (i > 1000)) {
	 args.error("GenColorMap nlevels out of range `"+args[2]+"'");
	 return;
      }
      reset_vars();
      TCL::execute(id+" repaint");
   } else if (args[1] == "movekey") {
      if (args.count() != 4) {
	 args.error("GenColorMap needs index ratio ");
	 return;
      }
      if (!args[2].get_int(i)) {
	 args.error("GenColorMap can't parse index `"+args[2]+"'");
	 return;
      }
      if ((i < 0) || (i > keys.size())) {
	 args.error("GenColorMap index out of range `"+args[2]+"'");
	 return;
      }
      if (!args[3].get_double(f)) {
	 args.error("GenColorMap can't parse ratio `"+args[3]+"'");
	 return;
      }
      if ((f < 0) || (f > 1)) {
	 args.error("GenColorMap ratio out of range `"+args[3]+"'");
	 return;
      }
      keys[i].ratio = f;
      spline[i].ratio = f;
   } else if (args[1] == "addkey") {
      if (args.count() != 4) {
	 args.error("GenColorMap needs ratio index");
	 return;
      }
      if (!args[2].get_double(f)) {
	 args.error("GenColorMap can't parse ratio `"+args[2]+"'");
	 return;
      }
      if ((f < 0) || (f > 1)) {
	 args.error("GenColorMap ratio out of range `"+args[2]+"'");
	 return;
      }
      if (!args[3].get_int(i)) {
	 args.error("GenColorMap can't parse index `"+args[3]+"'");
	 return;
      }
      if ((i < 0) || (i > keys.size())) {
	 args.error("GenColorMap index out of range `"+args[3]+"'");
	 return;
      }
      MaterialKey newmat;
      if (making_map==0) newmat = find_level(f);
      newmat.ratio = f;
      keys.insert(i, newmat);
      spline.insertData(i, newmat);
   } else if (args[1] == "delkey") {
      if (args.count() != 3) {
	 args.error("GenColorMap needs index");
	 return;
      }
      if (!args[2].get_int(i)) {
	 args.error("GenColorMap can't parse index `"+args[2]+"'");
	 return;
      }
      if ((i <= 0) || (i >= keys.size()-1)) {
	 args.error("GenColorMap index out of range `"+args[2]+"'");
	 return;
      }
      keys.remove(i);
      spline.removeData(i);
   } else if (args[1] == "cinterp") {
      if (args.count() != 2) {
	 args.error("GenColorMap needs no params");
	 return;
      }
      reset_vars();
      int hsv;
      if (cinterp_type.get() == "HSV") hsv = 1;
      else hsv = 0;
      for (i=0; i<keys.size(); i++) {
	 spline[i].hsv = keys[i].hsv = hsv;
      }
   } else if (args[1] == "interp") {
      if (args.count() != 2) {
	 args.error("GenColorMap needs no params");
	 return;
      }
      reset_vars();
   } else if (args[1] == "setmap") {
      if (args.count() != 3) {
	 args.error("GenColorMap needs no params");
	 return;
      }
      reset_vars();
      gen_map(args[2]);
   } else {
      Module::tcl_command(args, userdata);
   }
}


void
GenColorMap::execute()
{
   int nl(nlevels.get()-1);
   double minval=min.get();
   double maxval=max.get();
   ColorMapHandle cmap;
   if(gen_rangemap.get()){
       nl=2;
       cmap=scinew ColorMap(3, minval, maxval, 1);
       cmap->colors[0]=scinew Material(Color(0,0,0), Color(0,0,1), Color(0,0,0), 0);
       cmap->colors[1]=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
       cmap->colors[2]=scinew Material(Color(0,0,0), Color(1,0,0), Color(0,0,0), 0);
   } else {
     cmap=scinew ColorMap(nl+1, minval, maxval);
     for(int i=0;i<=nl;i++){
       cmap->colors[i]=scinew Material(find_level(i/(double)nl));
     }
   }

   // now see if diffuse is the only varying parameter...

   for (int i=1;i<=nl;i++) {
     if ((cmap->colors[i]->ambient != cmap->colors[0]->ambient) ||
	 (cmap->colors[i]->specular != cmap->colors[0]->specular) ||
	 (cmap->colors[i]->shininess != cmap->colors[0]->shininess) ||
	 (cmap->colors[i]->emission != cmap->colors[0]->emission) ||
	 (cmap->colors[i]->reflectivity != cmap->colors[0]->reflectivity) ||
	 (cmap->colors[i]->transparency != cmap->colors[0]->transparency) ||
	 (cmap->colors[i]->refraction_index != cmap->colors[0]->refraction_index))
       {
	 i = nl+2; // break out of the loop if it isn't constant
       }
	 
   }

   // set to 1 if it was constant (set to 0 in constructor)
   if (i == (nl+1))
     cmap->non_diffuse_constant = 1;

   outport->send(cmap);
}

Material
GenColorMap::find_level( const double f )
{
   int ks(keys.size());
   int idx1(0), idx2(ks-1);
   ASSERT((f>=0)&&(f<=1));
   clString it(interp_type.get());
   if (it == "Linear") {
      int i2;
      for (i2=1; i2<=(ks-1); i2++) {
	 if (keys[i2].ratio <= f) {
	    idx1 = i2;
	 }
      }
      for (i2=ks-2; i2>=0; i2--) {
	 if (keys[i2].ratio >= f) {
	    idx2 = i2;
	 }
      }
      if (keys[idx1].ratio == keys[idx2].ratio)
	 return keys[idx1];
      double t=(f-keys[idx1].ratio)/(keys[idx2].ratio-keys[idx1].ratio);
      return keys[idx1]*(1-t) + keys[idx2]*t;
   } else if (it == "Spline") {
      Material m(spline(f));
      if (m.ambient.r() < 0) m.ambient = Color(0,m.ambient.g(),m.ambient.b());
      if (m.ambient.r() > 1) m.ambient = Color(1,m.ambient.g(),m.ambient.b());
      if (m.ambient.g() < 0) m.ambient = Color(m.ambient.r(),0,m.ambient.b());
      if (m.ambient.g() > 1) m.ambient = Color(m.ambient.r(),1,m.ambient.b());
      if (m.ambient.b() < 0) m.ambient = Color(m.ambient.r(),m.ambient.g(),0);
      if (m.ambient.b() > 1) m.ambient = Color(m.ambient.r(),m.ambient.g(),1);
      if (m.diffuse.r() < 0) m.diffuse = Color(0,m.diffuse.g(),m.diffuse.b());
      if (m.diffuse.r() > 1) m.diffuse = Color(1,m.diffuse.g(),m.diffuse.b());
      if (m.diffuse.g() < 0) m.diffuse = Color(m.diffuse.r(),0,m.diffuse.b());
      if (m.diffuse.g() > 1) m.diffuse = Color(m.diffuse.r(),1,m.diffuse.b());
      if (m.diffuse.b() < 0) m.diffuse = Color(m.diffuse.r(),m.diffuse.g(),0);
      if (m.diffuse.b() > 1) m.diffuse = Color(m.diffuse.r(),m.diffuse.g(),1);
      if (m.specular.r() < 0) m.specular = Color(0,m.specular.g(),m.specular.b());
      if (m.specular.r() > 1) m.specular = Color(1,m.specular.g(),m.specular.b());
      if (m.specular.g() < 0) m.specular = Color(m.specular.r(),0,m.specular.b());
      if (m.specular.g() > 1) m.specular = Color(m.specular.r(),1,m.specular.b());
      if (m.specular.b() < 0) m.specular = Color(m.specular.r(),m.specular.g(),0);
      if (m.specular.b() > 1) m.specular = Color(m.specular.r(),m.specular.g(),1);
      if (m.emission.r() < 0) m.emission = Color(0,m.emission.g(),m.emission.b());
      if (m.emission.r() > 1) m.emission = Color(1,m.emission.g(),m.emission.b());
      if (m.emission.g() < 0) m.emission = Color(m.emission.r(),0,m.emission.b());
      if (m.emission.g() > 1) m.emission = Color(m.emission.r(),1,m.emission.b());
      if (m.emission.b() < 0) m.emission = Color(m.emission.r(),m.emission.g(),0);
      if (m.emission.b() > 1) m.emission = Color(m.emission.r(),m.emission.g(),1);
      if (m.reflectivity < 0) m.reflectivity = 0;
      if (m.reflectivity > 1) m.reflectivity = 1;
      if (m.transparency < 0) m.transparency = 0;
      if (m.transparency > 1) m.transparency = 1;
      return m;
   } else {
      error("Unknown interpolation type!");
      return Material();
   }    
}

void
GenColorMap::gen_map( const clString& mt )
{
   MaterialKey k0(0);
   MaterialKey k1(0.5);
   MaterialKey k2(1);
   Color black(0,0,0);
   Color white(1,1,1);
   Color red(1,0,0);
   Color green(0,1,0);
   Color blue(0,0,1);

   making_map = 1; // To ensure that find_level/repaint are NOT called.
   
   TCL::execute(id+" clearkeys"); // clears all keys but 0,1
   keys.remove_all();
      
   if (mt == "Rainbow") {
      cinterp_type.set("HSV");
      interp_type.set("Linear");
      k0.diffuse = HSVColor(1,1,1);
      k2.diffuse = HSVColor(300,1,1);
      keys.add(k0);
      keys.add(k2);
   } else if (mt == "Voltage") {
      cinterp_type.set("RGB");
      interp_type.set("Linear");
      k0.diffuse = blue;
      k1.diffuse = green;
      k2.diffuse = red;
      keys.add(k0);
      keys.add(k2);
      TCL::execute(id+" addkey 0.5"); // adds to C++ also.
      keys[1] = k1;
   } else if (mt == "Voltage2") {
      cinterp_type.set("RGB");
      interp_type.set("Linear");
      k1.ratio=.48;
      k2.ratio=.5;
      MaterialKey k3(.52);
      MaterialKey k4(1.0);
      k0.diffuse = HSVColor(240,1,0.5);
      k1.diffuse = HSVColor(240,1,1);
      k2.diffuse = HSVColor(120,1,1);
      k3.diffuse = HSVColor(0,.8,.85);
      k4.diffuse = HSVColor(0,1,0.5);
      keys.add(k0);
      keys.add(k4);
      TCL::execute(id+" addkey 0.48"); // adds to C++ also.
      keys[1] = k1;
      TCL::execute(id+" addkey 0.5"); // adds to C++ also.
      keys[2] = k2; 
      TCL::execute(id+" addkey 0.52"); // adds to C++ also.
      keys[3] = k3;
   } else if (mt == "RedToGreen") {
      cinterp_type.set("RGB");
      interp_type.set("Linear");
      k0.diffuse = red;
      k2.diffuse = green;
      keys.add(k0);
      keys.add(k2);
   } else if (mt == "RedToBlue") {
      cinterp_type.set("RGB");
      interp_type.set("Linear");
      k0.diffuse = red;
      k2.diffuse = blue;
      keys.add(k0);
      keys.add(k2);
   } else if (mt == "Grayscale") {
      cinterp_type.set("RGB");
      interp_type.set("Linear");
      k0.diffuse = black;
      k2.diffuse = white;
      keys.add(k0);
      keys.add(k2);
   } else if (mt == "Red") {
      cinterp_type.set("RGB");
      interp_type.set("Linear");
      k0.diffuse = black;
      k2.diffuse = red;
      keys.add(k0);
      keys.add(k2);
   } else if (mt == "Spline") {
      cinterp_type.set("RGB");
      interp_type.set("Spline");
      MaterialKey k3(1);
      k1.ratio = 1.0/3.0;
      k2.ratio = 2.0/3.0;
      k0.diffuse = blue;
      k1.diffuse = green;
      k2.diffuse = red;
      k3.diffuse = white;
      keys.add(k0);
      keys.add(k3);
      TCL::execute(id+" addkey "+to_string(1.0/3.0)); // adds to C++ also.
      keys[1] = k1;
      TCL::execute(id+" addkey "+to_string(2.0/3.0)); // adds to C++ also.
      keys[2] = k2;
   } else {
      cerr << "Unknown map type!" << endl;
   }
   making_map = 0;

   int hsv;
   if (cinterp_type.get() == "HSV") hsv=1;
   else hsv=0;
   for (int i=0; i<keys.size(); i++)
      keys[i].hsv = hsv;

   spline.setData(keys);
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<MaterialKey>;
template class Array1<clString>;

#include <Math/CatmullRomSpline.cc>
template class CatmullRomSpline<MaterialKey>;

#endif
