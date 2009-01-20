#ifndef VolElement_H
#define VolElement_H

class VolElement {
public:
  VolElement();
  void get_limits(double *VolTable, const int &vIndex);
  void get_public_limits(double &_xlow, double &_xup,
			 double &_ylow, double &_yup,
			 double &_zlow, double &_zup);  
  ~VolElement();
private:
  double xlow, xup, ylow, yup, zlow, zup;
};

#endif
