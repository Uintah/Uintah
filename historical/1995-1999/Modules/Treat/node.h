#ifndef _NODE_    /* include this file 1 time only */

#define _NODE_

#define Node tNode

class Node
{
public:
  void setNum(int val) { num = val;}
  void setX(double val) { x = val; }
  void setY(double val) { y = val; }
  void setZ(double val) { z = val; }
  void setVX(double val) { vx = val; }
  void setVY(double val) { vy = val; }
  void setVZ(double val) { vz = val; }
  void setSarc(double val) { sarc = val; }
  void setBoundType(int i) { boundType = i; }
  void setBoundVal(double i) { boundVal = i; }
  void setTemp(double val) {temp = val; }
  void setTempPast(double val) {tempPast = val;}
  void setRdof(int val) {rdof = val; }
  int getNum() const { return num; }
  double getX() const { return x; }
  double getY() const { return y; }
  double getZ() const { return z; }
  double getVX() const { return vx; }
  double getVY() const { return vy; }
  double getVZ() const { return vz; }
  double getSarc() const { return sarc; }
  int getBoundType() const { return boundType; }
  double getBoundVal() const { return boundVal; }
  double getTemp() const { return temp; }
  double getTempPast() const { return tempPast; }
  int getRdof() { return rdof; }
  static int numberOfBCS;
  
private:
  int num;
  double x;
  double y;
  double z;
  double vx;
  double vy;
  double vz;
  double sarc;
  int boundType;
  double boundVal;
  double temp;
  double tempPast;
  int rdof;
  
};

#endif


