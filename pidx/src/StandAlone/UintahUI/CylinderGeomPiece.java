//**************************************************************************
// Class   : CylinderGeomPiece
// Purpose : Creates a cylinder
//**************************************************************************
import java.lang.System;
import java.io.PrintWriter;

public class CylinderGeomPiece extends GeomPiece {

  // Common data
  private Point d_bottom = null;
  private Point d_top = null;
  private double d_radius = 0.0;

  public CylinderGeomPiece() {
    d_name = new String("Cylinder");
    d_bottom = new Point();
    d_top = new Point();
    d_radius = 0.0;
  }
  
  public CylinderGeomPiece(String name, Point center, double radius, 
                            double length) {
    d_name = new String(name);
    d_radius = radius;
    d_bottom = new Point(center);
    d_top = new Point(center.getX(), center.getY(), center.getZ()+length);
  }
  
  public String getName() {
   return d_name;
  }
  
  // Common Methods
  public void writeUintah(PrintWriter pw, String tab){

    String tab1 = new String(tab+"  ");
    pw.println(tab+"<cylinder label=\""+d_name+"\">");
    pw.println(tab1+"<bottom> ["+d_bottom.getX()+", "+d_bottom.getY()+", "+
               d_bottom.getZ()+"] </bottom>");
    pw.println(tab1+"<top> ["+d_top.getX()+", "+d_top.getY()+", "+
               d_top.getZ()+"] </top>");
    pw.println(tab1+"<radius> "+d_radius+" </radius>");
    pw.println(tab+"</cylinder>");
  }

  public void print(){
    String tab1 = new String("  ");
    System.out.println("<cylinder label=\""+d_name+"\">");
    System.out.println(tab1+"<bottom> ["+d_bottom.getX()+", "+
               d_bottom.getY()+", "+ d_bottom.getZ()+"] </bottom>");
    System.out.println(tab1+"<top> ["+d_top.getX()+", "+
               d_top.getY()+", "+ d_top.getZ()+"] </top>");
    System.out.println(tab1+"<radius> "+d_radius+" </radius>");
    System.out.println("</cylinder>");
  }


}
