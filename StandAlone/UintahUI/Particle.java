//**************************************************************************
// Program : Particle.java
// Purpose : Define the Particle objects
// Author  : Biswajit Banerjee
// Date    : 03/24/1999
// Mods    :
//**************************************************************************

import java.io.*;

//**************************************************************************
// Class   : Particle
// Purpose : Creates a Particle object
//**************************************************************************
public class Particle extends Object {

  // Data
  private double d_radius = 0.0;
  private double d_length = 1.0;
  private Point d_center = null;
  private int d_matCode = 0;
  private int d_type = 1;
  private double d_rotation = 0.0;

  // Public static data
  static final int CIRCLE = 1;
  static final int SPHERE = 2;

  // Constructor
  public Particle() {
    d_radius = 0.0;
    d_length = 1.0;
    d_center = new Point(0.0,0.0,0.0);
    d_matCode = 0;
    d_type = CIRCLE;
    d_rotation = 0.0;
  }

  public Particle(int type) {
    d_radius = 0.0;
    d_length = 1.0;
    d_center = new Point(0.0,0.0,0.0);
    d_matCode = 0;
    d_type = type;
    d_rotation = 0.0;
  }

  public Particle(double radius, double length, Point center, int matCode) {
    d_radius = radius;
    d_length = length;
    d_center = new Point(center);
    d_matCode = matCode;
    d_type = CIRCLE;
    d_rotation = 0.0;
  }

  public Particle(int type, double radius, double rotation, Point center,
		  int matCode) {
    d_type = type;
    d_radius = radius;
    d_rotation = rotation;
    d_center = new Point(center);
    d_matCode = matCode;
    d_length = 0.0;
  }

  // Get Particle data
  public double getRadius() {return d_radius;}
  public double getLength() {return d_length;}
  public Point getCenter() {return d_center;}
  public int getMatCode() {return d_matCode;}
  public int getType() {return d_type;}
  public double getRotation() {return d_rotation;}

  // Set Particles data
  public void setRadius(double radius) { d_radius = radius;}
  public void setLength(double length) { d_length = length;}
  public void setCenter(Point center) { d_center = center;}
  public void setMatCode(int matCode) { d_matCode = matCode;}
  public void setType(int type) {d_type = type;}
  public void setRotation(double rotation) {d_rotation = rotation;}

  // Print the particle data
  public void print() 
  {
    System.out.println("Material Code = "+d_matCode+" Type = "+d_type+
                       " Rad = "+d_radius+" Length = "+d_length+
                       " Rotation = "+d_rotation+ " Center = ["+
                       d_center.getX()+", "+d_center.getY()+", "+
                       d_center.getZ()+"]"); 
  }
}
