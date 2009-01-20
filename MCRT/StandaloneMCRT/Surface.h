#ifndef Surface_H
#define Surface_H

#include <cmath>

class RealSurface;
class VirtualSurface;

class Surface {
public:
  Surface();
  virtual ~Surface();
  
  void getPhi(const double &random);
  virtual void getTheta(const double &random) = 0;
  
  friend class RealSurface;
  friend class VirtualSurface;
  
protected:
  double theta, phi;
  
};

#endif
  
  
