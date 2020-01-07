//**************************************************************************
// Class   : DifferenceGeomPiece
// Purpose : Creates a box
//**************************************************************************

import java.lang.System;
import java.io.PrintWriter;

public class DifferenceGeomPiece extends GeomPiece {

  // Common data
  private GeomPiece d_geomPiece1 = null;
  private GeomPiece d_geomPiece2 = null;

  public DifferenceGeomPiece(String name, GeomPiece geomPiece1, GeomPiece geomPiece2) {

    if (geomPiece1 == null || geomPiece2 == null) return;

    d_name = name;
    d_geomPiece1 = geomPiece1;
    d_geomPiece2 = geomPiece2;
  }
  
  public String getName() {
   return d_name;
  }
  
  // Print
  public void writeUintah(PrintWriter pw, String tab) {

    if (d_geomPiece1 == null || d_geomPiece2 == null) return;

    String tab1 = new String(tab+"  ");
    pw.println(tab+"<difference label=\""+d_name+"\">");
    d_geomPiece1.writeUintah(pw, tab1);
    d_geomPiece2.writeUintah(pw, tab1);
    pw.println(tab+"</difference>");
  }

  public void print() {

    if (d_geomPiece1 == null || d_geomPiece2 == null) return;

    System.out.println("<difference label=\""+d_name+"\">");
    d_geomPiece1.print();
    d_geomPiece2.print();
    System.out.println("</difference>");

  }
}
