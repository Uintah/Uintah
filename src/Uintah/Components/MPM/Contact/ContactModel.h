class ContactModel {
 public:
  void computeSurfaceNormals();
  void computeTractions();
  virtual void exchangeMomentumInterpolated() = 0;
  virtual void exchangeMomentumIntegrated() = 0;
 

}
