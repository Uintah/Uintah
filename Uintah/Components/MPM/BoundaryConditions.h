class BoundaryConditions {
 public:
  virtual void loadBoundaryConditions() = 0;
  virtual void displacementBoundaryConditions() = 0;
  virtual void heatFluxBoundaryConditions() = 0;
};
