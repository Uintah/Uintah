//**************************************************************************
// Program : GeomObject.java
// Purpose : Define the Geometry objects
// Author  : Biswajit Banerjee
// Date    : 05/14/2006
// Mods    :
//**************************************************************************

import java.util.Vector;
import java.io.PrintWriter;

//**************************************************************************
// Class   : GeomObject
// Purpose : Creates a GeomObject 
//**************************************************************************
public class GeomObject extends Object {

  // Common data
  private String d_name;
  private Vector3D d_resolution;
  private Vector3D d_velocity;
  private double d_temperature;
  private double d_density;
  private double d_pressure;
  private Vector d_geomPieceVector;
  
  // Constructor
  public GeomObject() {
    d_name = new String("Default");
    d_resolution = new Vector3D();
    d_velocity = new Vector3D();
    d_temperature = 0.0;
    d_density = 0.0;
    d_pressure = 0.0;
    d_geomPieceVector = new Vector();
  }

  // Get/Set the data
  public String getName() {
    return d_name;
  }

  public void setName(String name) {
    d_name = name;
  }

  public void setResolution(double x, double y, double z) {
    d_resolution.set(x, y, z);
  }

  public void setVelocity(double x, double y, double z) {
    d_velocity.set(x, y, z);
  }

  public void setPressure(double pressure) {
    d_pressure = pressure;
  }

  public void setTemperature(double temperature) {
    d_temperature = temperature;
  }

  public void setDensity(double density) {
    d_density = density;
  }

  public void addGeomPiece(GeomPiece geomPiece) {
    d_geomPieceVector.addElement(geomPiece);
  }

  public void removeGeomPieceAt(int index) {
    d_geomPieceVector.removeElementAt(index);
  }

  // Write the geometry object in Uintah format
  public void writeUintah(PrintWriter pw, String tab){

    String tab1 = new String(tab+"  ");
    pw.println(tab+"<geom_object>");
    pw.println(tab1+"<res> ["+(int)d_resolution.x()+", "+
       (int) d_resolution.y()+", "+ (int) d_resolution.z()+"] </res>");
    pw.println(tab1+"<velocity> ["+d_velocity.x()+", "+
       d_velocity.y()+", "+d_velocity.z()+"] </velocity>");
    pw.println(tab1+"<temperature> "+d_temperature+" </temperature>");
    if (d_density > 0.0) {
      pw.println(tab1+"<density> "+d_density+" </density>");
      pw.println(tab1+"<pressure> "+d_pressure+" </pressure>");
    }
    
    for (int ii=0; ii < d_geomPieceVector.size(); ++ii) {
      GeomPiece geomPiece = (GeomPiece) d_geomPieceVector.elementAt(ii);
      geomPiece.writeUintah(pw, tab1);
    }

    pw.println(tab+"</geom_object>");
  }

  public void print(){

    String tab1 = new String("  ");
    System.out.println("<geom_object>");
    System.out.println(tab1+"<res> ["+(int)d_resolution.x()+", "+
       (int) d_resolution.y()+", "+ (int) d_resolution.z()+"] </res>");
    System.out.println(tab1+"<velocity> ["+d_velocity.x()+", "+
       d_velocity.y()+", "+d_velocity.z()+"] </velocity>");
    System.out.println(tab1+"<temperature> "+d_temperature+" </temperature>");
    System.out.println(tab1+"<density> "+d_density+" </density>");
    System.out.println(tab1+"<pressure> "+d_density+" </pressure>");
    
    for (int ii=0; ii < d_geomPieceVector.size(); ++ii) {
      GeomPiece geomPiece = (GeomPiece) d_geomPieceVector.elementAt(ii);
      geomPiece.print();
    }

    System.out.println("</geom_object>");
  }

}
