/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



/****************************************************************************\
 *                                                                          *
 * filename: TimeVaryingInstance.h                                          *
 * author  : R. Keith Morley                                                *
 * last mod: 07/12/02                                                       *
 *                                                                          *
\****************************************************************************/ 

#ifndef _TIME_VARYING_INSTANCE_
#define _TIME_VARYING_INSTANCE_ 1


#include <Packages/rtrt/Core/Instance.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/BBox.h>

namespace rtrt {
class TimeVaryingInstance;
class FishInstance1;
class FishInstance2;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::TimeVaryingInstance*&);
  void Pio(Piostream&, rtrt::FishInstance1*&);
  void Pio(Piostream&, rtrt::FishInstance2*&);
}

namespace rtrt 
{

/***************************************************************************/
class TimeVaryingInstance: public Instance
{
public:
  Point center;
  Point ocenter;
  Vector axis;
  double rate;
  BBox bbox_orig;
  Vector originToCenter; 
  TimeVaryingInstance (InstanceWrapperObject* obj);
  TimeVaryingInstance() : Instance() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TimeVaryingInstance*&);

  virtual void computeTransform(double t); 
  virtual void compute_bounds(BBox& b, double /*offset*/);
  virtual void animate(double t, bool& changed);
  void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		 PerProcessorContext* ppc);
}; // TimeVaryingInstance

/****************************************************************************/
class FishInstance1 : public TimeVaryingInstance
{
public:   
  double vertHeightScale;
  double horizHeightScale;
  double vertPerScale;
  double horizPerScale;
  double rotPerSec;
  double startTime;
  double vertShift;
   
  FishInstance1(InstanceWrapperObject* obj,
                double _vertHeightScale, double _horizHeightScale, 
		double _vertPerScale, double _horizPerScale,   
		double _rotPerSec, double _startTime, 
		double _vertShift);
  FishInstance1() : TimeVaryingInstance() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, FishInstance1*&);

  virtual void computeTransform(double t);
}; // FishInstance1

/****************************************************************************/
class FishInstance2 : public TimeVaryingInstance
{
public:   
  double vertHeightScale;
  double horizHeightScale;
  double vertPerScale;
  double horizPerScale;
  double rotPerSec;
  double startTime;
  double vertShift;

  FishInstance2(InstanceWrapperObject* obj,
                double _vertHeightScale, double _horizHeightScale, 
		double _vertPerScale, double _horizPerScale,   
		double _rotPerSec, double _startTime, 
		double _vertShift);

  FishInstance2() : TimeVaryingInstance() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, FishInstance2*&);

  virtual void computeTransform(double t);
}; // FishInstance2

}  // rtrt

#endif // _TIME_VARYING_INSTANCE_

