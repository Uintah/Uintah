/*************************************************
** TimeVaryingCheapCaustics.h                   **
** --------------------------                   **
**                                              **
** This class loads in an array of binary PGM   **
** files, and allows indexing into the "time-   **
** varying" data.   Obviously, to use this,     **
** RTRT materials need to be aware of this      **
** object, and call the class functions.  This  **
** object is designed to replace a light, so    **
** that after shooting the shadow rays instead  **
** of using a light color, a value from this    **
** array can be substituted.                    **
**                                              **
** Thie idea is we are projecting               **
** (orthographically) precomputed caustics data **
** onto the scene.                              **
**                                              **
** Chris Wyman (6/24/2002)                      **
*************************************************/

#ifndef TimeVaryingCheapCaustics_H
#define TimeVaryingCheapCaustics_H 1

#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Core/Geometry/Point.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Color.h>
#include <iostream>

namespace rtrt {
  class TimeVaryingCheapCaustics;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::TimeVaryingCheapCaustics*&);
}

namespace rtrt {

class TimeVaryingCheapCaustics : public SCIRun::Persistent {
  // The number of files to load
  int numFiles;
  // The array of caustics data.  
  Array1< Array2< float > * > caustics;
  // If one frame in the above data != 1 second, multiply by this scale
  float timeScale;
  // If the "rectangle" we stretch the data over needs more than 
  // one copy of the caustics, tile it (using this factor)
  float tilingFactor;
  // The current time.  This can be overridden in the GetCausticColor() call
  float currentTime;
  // Since the input is a PGM, maybe we want an aqua color light.  Or something.
  Color lightColor;
  // The corner of the rectangle to project the caustics from
  SCIRun::Point minPoint;
  // the u & v axes of the rectangle (normalized)
  SCIRun::Vector uAxis, vAxis;
  // the original lengths of the u & v vectors
  float uVecLen, vVecLen;
  // the u & v dimenstions of the maps
  int uDim, vDim;
  // the axis we project from
  SCIRun::Vector projAxis;

  // internal data
  float totalTime;
  float uSizePerTile, vSizePerTile;

public:
  // fileNames should be of the form "caustic%d.pgm" (e.g. can be used in
  //    sprintf to get filenames)
  // numFiles is how many timesteps are in the dataset (e.g. 32 means
  //    caustic0.pgm-caustic31.pgm contain the data)
  TimeVaryingCheapCaustics( char *fileNames, int numFiles, 
			    SCIRun::Point minPoint, SCIRun::Vector uAxis, SCIRun::Vector vAxis,
			    Color lightColor=Color(1,1,1),
			    float timeScale=1.0, float tilingFactor=1.0 );
  ~TimeVaryingCheapCaustics();

  TimeVaryingCheapCaustics() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TimeVaryingCheapCaustics*&);

  // sets the current time. 
  inline void SetCurrentTime( float newTime ) { currentTime = newTime; }
  // gets the projection axis, so you don't need to hardcode it into materials.
  inline SCIRun::Vector GetProjectionAxis() { return projAxis; }
  // gets the caustic's color at atPoint, at currTime.  if currTime < 0,
  //   use the currentTime stored in the class.
  Color GetCausticColor( SCIRun::Point atPoint, float currTime=-1 );
};

}

#endif
