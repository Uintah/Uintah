#ifndef _PROP_    /* include this file 1 time only */

#define _PROP_


class Property {

public:
  Property();
  ~Property();
  void setSS(int mySS) { ss = mySS; }
  void setSymmetric(int mySymmetric) { symmetric = mySymmetric; }
  void setTsubA(double myTsubA) {tsuba = myTsubA; }
  void setTheta(double myTheta) {theta = myTheta; }
  void setDeltaT(double myDeltaT) {deltaT = myDeltaT; }
  void setTimeSteps(int myTimeSteps) {timeSteps = myTimeSteps; }
  void setWriteEvery(int myWriteEvery) {writeEvery = myWriteEvery; }
  void setPeriod(int myPeriod) {period = myPeriod; }
  void setEPS(double myEPS) { eps = myEPS; }
  int getSS() { return ss; }
  int getSymmetric() { return symmetric; }
  double getTsubA() { return tsuba; }
  double getTheta() { return theta; }
  double getDeltaT() { return deltaT; }
  int getTimeSteps() { return timeSteps; }
  int getWriteEvery() { return writeEvery; }
  int getPeriod() { return period; }
  double getEPS() { return eps; }
  
private:
  
  int ss;
  int symmetric;
  double theta;
  double deltaT;
  double tsuba;
  int period;
  int timeSteps;
  int writeEvery;
  double eps;

};

#endif



