//**************************************************************************
// Program : Point.java
// Purpose : Define the Point objects
// Author  : Biswajit Banerjee
// Date    : 03/24/1999
// Mods    :
//**************************************************************************
// $Id: Point.java,v 1.2 2000/02/03 05:36:58 bbanerje Exp $

//**************************************************************************
// Class   : Point
// Purpose : Creates a Point object
//**************************************************************************
public class Point extends Object {

  // Data
  private double d_x = 0.0;
  private double d_y = 0.0;
  private double d_z = 0.0;

  // Constructor
  public Point() {
    d_x = 0.0;
    d_y = 0.0;
    d_z = 0.0;
  }

  public Point(double xCoord, double yCoord, double zCoord) {
    d_x = xCoord;
    d_y = yCoord;
    d_z = zCoord;
  }

  public Point(Point pt) {
    d_x = pt.d_x;
    d_y = pt.d_y;
    d_z = pt.d_z;
  }

  // Get Point data
  public double getX() {return d_x;}
  public double getY() {return d_y;}
  public double getZ() {return d_z;}

  // Set Points data
  public void setX(double val) { d_x = val;}
  public void setY(double val) { d_y = val;}
  public void setZ(double val) { d_z = val;}
}
// $Log: Point.java,v $
// Revision 1.2  2000/02/03 05:36:58  bbanerje
// Just a few changes in all the java files .. and some changes in
// GenerateParticleFrame and Particle and ParticleList
//
