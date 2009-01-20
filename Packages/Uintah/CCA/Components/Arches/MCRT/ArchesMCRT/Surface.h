#ifndef Surface_H
#define Surface_H

class Surface {
public:
  Surface();
  virtual ~Surface();
  
  void getPhi(double &phi, double &random);
  virtual void getTheta(double &theta, double &random) = 0;


};

#endif
  
  
