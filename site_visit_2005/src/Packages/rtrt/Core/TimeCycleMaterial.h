#ifndef TIMECYCLEMATERIAL_H
#define TIMECYCLEMATERIAL_H 1

#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {
class TimeCycleMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::TimeCycleMaterial*&);
}

namespace rtrt {

class TimeCycleMaterial : public CycleMaterial {

public:

  TimeCycleMaterial();
  virtual ~TimeCycleMaterial();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TimeCycleMaterial*&);

  void add( Material* mat, double time );
  //
  // Add material
  //
  // mat -- material to be added to a list
  // time -- amount of time for material to be displayed
  //
  virtual void shade( Color& result, const Ray& ray,
		      const HitInfo& hit, int depth, 
		      double atten, const Color& accumcolor,
		      Context* cx );

private:
    
  Array1<double> time_array_;
  double time_;
  double cur_time_;

};

} // end namespace rtrt

#endif
