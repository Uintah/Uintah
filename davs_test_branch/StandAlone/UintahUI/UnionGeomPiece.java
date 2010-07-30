//**************************************************************************
// Class   : UnionGeomPiece
// Purpose : Creates a box
//**************************************************************************

import java.lang.System;
import java.io.PrintWriter;
import java.util.Vector;

public class UnionGeomPiece extends GeomPiece {

  // Common data
  private Vector d_geomPiece = null;

  public UnionGeomPiece() {
    d_name = new String("Union");
    d_geomPiece = new Vector();
  }
  
  public UnionGeomPiece(String name) {
    d_name = new String(name);
    d_geomPiece = new Vector();
  }

  public UnionGeomPiece(String name, Vector geomPiece) {

    d_name = new String(name);
    if (geomPiece == null) {
      d_geomPiece = new Vector();
      return;
    }

    d_geomPiece = new Vector();
    int numGeomPiece = geomPiece.size();
    if (numGeomPiece > 0) {
      for (int ii = 0; ii < numGeomPiece; ++ii) {
        d_geomPiece.addElement((GeomPiece) geomPiece.elementAt(ii));
      }
    }
  }
  
  public void set(String name, Vector geomPiece) {

    d_name = name;
    if (geomPiece == null) return;

    d_geomPiece.clear();
    int numGeomPiece = geomPiece.size();
    if (numGeomPiece > 0) {
      for (int ii = 0; ii < numGeomPiece; ++ii) {
        d_geomPiece.addElement((GeomPiece) geomPiece.elementAt(ii));
      }
    }
  }

  public void addGeomPiece(GeomPiece geomPiece) {
    d_geomPiece.addElement(geomPiece);
  }
  
  public String getName() {
   return d_name;
  }
  
  // Print
  public void writeUintah(PrintWriter pw, String tab) {

    if (d_geomPiece == null) return;

    int numGeomPiece = d_geomPiece.size();
    if (numGeomPiece > 0) {

      String tab1 = new String(tab+"  ");
      pw.println(tab+"<union label=\""+d_name+"\">");

      for (int ii = 0; ii < numGeomPiece; ++ii) {
        GeomPiece geomPiece = (GeomPiece) d_geomPiece.elementAt(ii);
        geomPiece.writeUintah(pw, tab1);
      }
      pw.println(tab+"</union>");
    }
  }

  public void print() {

    if (d_geomPiece == null) return;

    int numGeomPiece = d_geomPiece.size();
    if (numGeomPiece > 0) {

      System.out.println("<union label=\""+d_name+"\">");

      for (int ii = 0; ii < numGeomPiece; ++ii) {
        GeomPiece geomPiece = (GeomPiece) d_geomPiece.elementAt(ii);
        geomPiece.print();
      }
      System.out.println("</union>");
    }
  }
}
