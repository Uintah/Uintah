#ifndef __FRACTURE_H__
#define __FRACTURE_H__

class Fracture {
 public:
  virtual void computeEnergyReleaseRate() = 0;
  virtual void computeCrackGrowthSpeed() = 0;
  virtual void computeCrackTiltingAngle() = 0;
  virtual void trackFracture() = 0;
  virtual bool crackTipCell(const Cell& cell) const = 0;

};


class TriangleSurfaceSplitter {
  virtual void splitTriangleSurfaces() = 0;
  virtual void mergeTriangleSurface() = 0;
};


#endif __FRACTURE_H__
