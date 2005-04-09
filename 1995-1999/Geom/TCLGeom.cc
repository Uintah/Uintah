
#include <Geom/TCLGeom.h>
#include <TCL/TCL.h>
#include <TCL/TCLTask.h>
#include <Geom/Color.h>
#include <Geom/Material.h>

TCLColor::TCLColor(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), r("r", str(), tcl), g("g", str(), tcl),
  b("b", str(), tcl)
{
}

TCLColor::~TCLColor()
{
}

Color TCLColor::get()
{
    return Color(r.get(), g.get(), b.get());
}

void TCLColor::set(const Color& p)
{
    r.set(p.r());
    g.set(p.g());
    b.set(p.b());
}

void TCLColor::emit(ostream& out)
{
    r.emit(out);
    g.emit(out);
    b.emit(out);
}

TCLMaterial::TCLMaterial(const clString& name, const clString& id, TCL* tcl)
: TCLvar(name, id, tcl), ambient("ambient", str(), tcl),
  diffuse("diffuse", str(), tcl), specular("specular", str(), tcl),
  shininess("shininess", str(), tcl), emission("emission", str(), tcl),
  reflectivity("reflectivity", str(), tcl),
  transparency("transparency", str(), tcl),
  refraction_index("refraction_index", str(), tcl)
{
}

TCLMaterial::~TCLMaterial()
{
}

Material TCLMaterial::get()
{
    Material m(ambient.get(), diffuse.get(), specular.get(), shininess.get());
    m.emission=emission.get();
    m.reflectivity=reflectivity.get();
    m.transparency=transparency.get();
    m.refraction_index=refraction_index.get();
    return m;
}

void TCLMaterial::set(const Material& m)
{
    ambient.set(m.ambient);
    diffuse.set(m.diffuse);
    specular.set(m.specular);
    shininess.set(m.shininess);
    emission.set(m.emission);
    reflectivity.set(m.reflectivity);
    transparency.set(m.transparency);
    refraction_index.set(m.refraction_index);
}

void TCLMaterial::emit(ostream& out)
{
    ambient.emit(out);
    diffuse.emit(out);
    specular.emit(out);
    shininess.emit(out);
    emission.emit(out);
    reflectivity.emit(out);
    transparency.emit(out);
    refraction_index.emit(out);
}
