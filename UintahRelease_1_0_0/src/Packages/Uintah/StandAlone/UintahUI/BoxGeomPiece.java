//**************************************************************************
// Class   : BoxGeomPiece
// Purpose : Creates a box
//**************************************************************************

import java.lang.System;
import java.io.PrintWriter;

public class BoxGeomPiece extends GeomPiece {

  // Common data
  private Point d_min = null;
  private Point d_max = null;

  public BoxGeomPiece() {
    d_name = new String("Box");
    d_min = new Point();
    d_max = new Point();
  }
  
  public BoxGeomPiece(String name, Point min, Point max) {
    d_name = new String(name);
    d_min = new Point(min);
    d_max = new Point(max);
  }
  
  public void set(String name, Point min, Point max) {
    d_name = name;
    d_min = min;
    d_max = max;
  }

  public String getName() {
   return d_name;
  }
  
  // Print
  public void writeUintah(PrintWriter pw, String tab) {

    String tab1 = new String(tab+"  ");
    pw.println(tab+"<box label=\""+d_name+"\">");
    pw.println(tab1+"<min> ["+d_min.getX()+", "+d_min.getY()+", "+
               d_min.getZ()+"] </min>");
    pw.println(tab1+"<max> ["+d_max.getX()+", "+d_max.getY()+", "+
               d_max.getZ()+"] </max>");
    pw.println(tab+"</box>");
  }

  public void print() {

    String tab1 = new String("  ");
    System.out.println("<box label=\""+d_name+"\">");
    System.out.println(tab1+"<min> ["+d_min.getX()+", "+d_min.getY()+", "+
               d_min.getZ()+"] </min>");
    System.out.println(tab1+"<max> ["+d_max.getX()+", "+d_max.getY()+", "+
               d_max.getZ()+"] </max>");
    System.out.println("</box>");
  }
}
