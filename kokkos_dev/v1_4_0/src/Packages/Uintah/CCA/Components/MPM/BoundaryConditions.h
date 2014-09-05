#ifndef __BOUNDARY_CONDITIONS_H__
#define __BOUNDARY_CONDITIONS_H__

class BoundaryConditions {
 public:
  virtual void loadBoundaryConditions() = 0;
  virtual void displacementBoundaryConditions() = 0;
  virtual void heatFluxBoundaryConditions() = 0;
};


#endif //__BOUNDARY_CONDITIONS_H__


