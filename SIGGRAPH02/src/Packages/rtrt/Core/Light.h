
#ifndef LIGHT_H
#define LIGHT_H 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Persistent/Persistent.h>

namespace rtrt {
  class Light;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Light*&);
}

namespace rtrt {

class Light : public SCIRun::Persistent {
public:

  std::string name_;
  double      radius;

  // default constructor for Pio only.
  Light() {}
  Light(const Point&, const Color&, double radius, double intensity = 1.0 );

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Light*&);

  inline const Point& get_pos() const { return pos; }
  inline const Color& get_color() const { return currentColor_; }
  inline       float  get_intensity() const { return intensity_; }
  inline Array1<Vector>& get_beamdirs() { return beamdirs; }

  void updateIntensity( double intensity );
  inline const Color& getOrigColor() const { return origColor_; }

  void setIndex(int index) { this->index=index; }
  int getIndex() const { return index; }

  inline bool isOn() { return isOn_; }
  inline void turnOn() { isOn_ = true; }
  inline void turnOff() { isOn_ = false; }

  void updatePosition( const Point & newPos );

  virtual Object *getSphere() const { return sphere_; }

protected:
  //! finish construction.
  void init();

  // Parameters for changing light value:
  float intensity_;
  Color currentColor_; // Original Color * intensity
  Color origColor_;

  Point pos;
  Color color;
  Array1<Vector> beamdirs;
  int index;

  bool isOn_;

  Sphere *sphere_; // sphere that will be rendered to represent this light.

};


} // end namespace rtrt

#endif
