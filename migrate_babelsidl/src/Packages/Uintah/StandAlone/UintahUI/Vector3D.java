//**************************************************************************
// Program : Vector3D.java
// Purpose : Define the Vector3D objects
// Author  : Biswajit Banerjee
// Date    : 05/14/2006
// Mods    :
//**************************************************************************

import java.io.PrintWriter;

//**************************************************************************
// Class   : Vector3D
// Purpose : Creates a three dimensional vector of doubles
//**************************************************************************
public class Vector3D extends Object {

  // Data
  private double d_x = 0.0;
  private double d_y = 0.0;
  private double d_z = 0.0;

  // Constructor
  public Vector3D() {
    d_x = 0.0;
    d_y = 0.0;
    d_z = 0.0;
  }

  public Vector3D(double x, double y, double z) {
    d_x = x;
    d_y = y;
    d_z = z;
  }

  public Vector3D(Vector3D vec) {
    d_x = vec.d_x;
    d_y = vec.d_y;
    d_z = vec.d_z;
  }

  // Get Vector3D data
  public double x() {return d_x;}
  public double y() {return d_y;}
  public double z() {return d_z;}

  // Set Vector3Ds data
  public void x(double val) { d_x = val;}
  public void y(double val) { d_y = val;}
  public void z(double val) { d_z = val;}

  public void set(Vector3D vec) {
    d_x = vec.d_x;
    d_y = vec.d_y;
    d_z = vec.d_z;
  }

  public void set(double x, double y, double z) {
    d_x = x;
    d_y = y;
    d_z = z;
  }

  public void add(Vector3D vec) {
    d_x += vec.d_x;
    d_y += vec.d_y;
    d_z += vec.d_z;
  }

  public double norm() {
    return (Math.sqrt(d_x*d_x+d_y*d_y+d_z*d_z));    
  }

  public double dot(Vector3D vec) {
    return (d_x*vec.d_x+d_y*vec.d_y+d_z*vec.d_z);    
  }

  public void print(PrintWriter pw) {
    pw.print("["+d_x+", "+d_y+", "+d_z+"]");
  }

  public void print() {
    System.out.print("["+d_x+", "+d_y+", "+d_z+"]");
  }
}
