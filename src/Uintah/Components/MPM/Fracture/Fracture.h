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
