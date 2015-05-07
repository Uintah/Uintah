//**************************************************************************
// Class   : SmoothCylGeomPiece
// Purpose : Creates a smooth cylinder
//**************************************************************************
import java.lang.System;
import java.io.PrintWriter;

public class SmoothCylGeomPiece extends GeomPiece {

  // Common data
  private Point d_bottom = null;
  private Point d_top = null;
  private double d_radius = 0.0;
  private double d_thickness = 0.0;
  private int d_numRadial = 0;
  private int d_numAxial = 0;
  private double d_arcStartAngle = 0.0;
  private double d_arcAngle = 0.0;

  public SmoothCylGeomPiece() {
    d_name = new String("SmoothCyl");
    d_bottom = new Point();
    d_top = new Point();
    d_radius = 0.0;
    d_thickness = 0.0;
    d_numRadial = 0;
    d_numAxial = 0;
    d_arcStartAngle = 0.0;
    d_arcAngle = 0.0;
  }
  
  public SmoothCylGeomPiece(String name, Point center, double radius, 
                            double thickness, double length) {
    d_name = new String(name);
    d_radius = radius;
    if (thickness == 0.0) {
      d_thickness = radius;
    } else {
      d_thickness = thickness;
    }
    d_numRadial = 0;
    d_numAxial = 0;
    d_arcStartAngle = 0.0;
    d_arcAngle = 360.0;
    d_bottom = new Point(center);
    d_top = new Point(center.getX(), center.getY(), center.getZ()+length);
  }
  
  public SmoothCylGeomPiece(String name, Point center, double radius, 
                            double thickness, double length, int numRadial,
                            int numAxial, double arcStart, double arcAngle ) {
    d_name = new String(name);
    d_radius = radius;
    if (thickness == 0.0) {
      d_thickness = radius;
    } else {
      d_thickness = thickness;
    }
    d_numRadial = numRadial;
    d_numAxial = numAxial;
    d_arcStartAngle = arcStart;
    d_arcAngle = arcAngle;
    d_bottom = new Point(center);
    d_top = new Point(center.getX(), center.getY(), center.getZ()+length);
  }
  
  public String getName() {
   return d_name;
  }
  
  // Common Methods
  public void writeUintah(PrintWriter pw, String tab){

    String tab1 = new String(tab+"  ");
    pw.println(tab+"<smoothcyl label=\""+d_name+"\">");
    pw.println(tab1+"<bottom> ["+d_bottom.getX()+", "+d_bottom.getY()+", "+
               d_bottom.getZ()+"] </bottom>");
    pw.println(tab1+"<top> ["+d_top.getX()+", "+d_top.getY()+", "+
               d_top.getZ()+"] </top>");
    pw.println(tab1+"<radius> "+d_radius+" </radius>");
    pw.println(tab1+"<thickness> "+d_thickness+" </thickness>");
    pw.println(tab1+"<num_radial> "+d_numRadial+" </num_radial>");
    pw.println(tab1+"<num_axial> "+d_numAxial+" </num_axial>");
    //pw.println(tab1+"<arc_start_angle> "+d_arcStartAngle+
    //                " </arc_start_angle>");
    //pw.println(tab1+"<arc_angle> "+d_arcAngle+ " </arc_angle>");
    pw.println(tab+"</smoothcyl>");
  }

  public void print(){
    String tab1 = new String("  ");
    System.out.println("<smoothcyl label=\""+d_name+"\">");
    System.out.println(tab1+"<bottom> ["+d_bottom.getX()+", "+
               d_bottom.getY()+", "+ d_bottom.getZ()+"] </bottom>");
    System.out.println(tab1+"<top> ["+d_top.getX()+", "+
               d_top.getY()+", "+ d_top.getZ()+"] </top>");
    System.out.println(tab1+"<radius> "+d_radius+" </radius>");
    System.out.println(tab1+"<thickness> "+d_thickness+" </thickness>");
    System.out.println(tab1+"<num_radial> "+d_numRadial+" </num_radial>");
    System.out.println(tab1+"<num_axial> "+d_numAxial+" </num_axial>");
    System.out.println(tab1+"<arc_start_angle> "+d_arcStartAngle+
                    " </arc_start_angle>");
    System.out.println(tab1+"<arc_angle> "+d_arcAngle+ " </arc_angle>");
    System.out.println("</smoothcyl>");
  }


}
