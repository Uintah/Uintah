//**************************************************************************
// Program : ParticleList.java
// Purpose : Define the ParticleList objects
// Author  : Biswajit Banerjee
// Date    : 03/24/1999
// Mods    :
//**************************************************************************
// $Id: ParticleList.java,v 1.2 2000/02/03 05:36:58 bbanerje Exp $

import java.io.*;
import java.util.*;

//**************************************************************************
// Class   : ParticleList
// Purpose : Creates a ParticleList object
//**************************************************************************
public class ParticleList extends Object {

  // Data
  private double d_rveSize = 0.0;
  private Vector d_particleList = null;
  private Vector d_triangleList = null;
  private Vector d_voronoiList = null;

  // Constructor
  public ParticleList() {
    d_rveSize = 100.0;
    d_particleList = new Vector();
    d_triangleList = new Vector();
    d_voronoiList = new Vector();
  }

  // Create a Particle list based on a vector of coordinates of interfaces
  public ParticleList(File particleFile) {
    d_particleList = new Vector();
    readFromFile(particleFile);
    d_triangleList = new Vector();
    d_voronoiList = new Vector();
  }

  // Read the particle data from file (for the new format - circles, squares,
  // spheres, cubes
  public void readFromFile(File particleFile) {
    d_particleList.clear();
    d_triangleList.clear();
    d_voronoiList.clear();
    try {
      FileReader fr = new FileReader(particleFile);
      StreamTokenizer st = new StreamTokenizer(fr);
      st.commentChar('#');
      st.parseNumbers();
      st.eolIsSignificant(true);
      double rveSize = 0.0;
      boolean first = true;
      int count = 0;
      int type = Particle.CIRCLE;
      double radius = 0.0;
      double rotation = 0.0;
      double xx = 0.0;
      double yy = 0.0;
      double zz = 0.0;
      int matCode = 0;
      int ttval = 0;
      while((ttval = st.nextToken()) != StreamTokenizer.TT_EOF) {
        if (first) {
          if (ttval == StreamTokenizer.TT_NUMBER) {
            d_rveSize = st.nval;
            first = false;
          }
        } else {
          if (ttval == StreamTokenizer.TT_NUMBER) {
            ++count;
            double ii = st.nval;
            switch (count) {
              case 1: type = (int) ii; break;
              case 2: radius = ii; break;
              case 3: rotation = ii; break;
              case 4: xx = ii; break;
              case 5: yy = ii; break;
              case 6: zz = ii; break;
              case 7: matCode = (int) ii; break;
              default: break;
            }
          }
          if (ttval == StreamTokenizer.TT_EOL && count != 0) {
            //System.out.println(type+" "+radius+" "+rotation+" "+xx+" "+yy+
                  //           " "+zz+" "+matCode);
            Point center = new Point(xx, yy, zz);
            Particle particle = new Particle(type, radius, rotation, 
                                             center, matCode);
            this.addParticle(particle);
            count = 0;
          }
        }
      }
    } catch (Exception e) {
      System.out.println("Could not read from "+particleFile.getName());
    }
  }

  // Write the particle data in Uintah XML format
  public void writeUintah(PrintWriter pw) {

    if (pw == null) return;

    int nofParts = size();
    for (int ii = 0; ii < nofParts; ii++) {
      Particle part = getParticle(ii);
      double rad = part.getRadius();
      double x = part.getCenter().getX();
      double y = part.getCenter().getY();
      pw.println("cyl4,"+x*1000.0+","+y*1000.0+","+rad*1000.0);
    }
  }

  // Save the particle data to file (for the new format - circles, squares,
  // spheres, cubes)
  public void saveToFile(File particleFile, int partType) {
    try {
      
      // Create a filewriter and the associated printwriter
      FileWriter fw = new FileWriter(particleFile);
      PrintWriter pw = new PrintWriter(fw);

      // Write the data
      pw.println("# RVE Size");
      pw.println(d_rveSize);
      pw.println("# Particle List");
      pw.println("# type  radius  rotation  xCent  yCent  zCent  matCode");
      int nofParts = size();
      for (int ii = 0; ii < nofParts; ii++) {
        Particle part = getParticle(ii);
        double radius = part.getRadius();
        double rotation = part.getRotation();
        double xCent = part.getCenter().getX();
        double yCent = part.getCenter().getY();
        double zCent = part.getCenter().getZ();
        int matCode = part.getMatCode();
        pw.println("# Particle "+ii);
        pw.println(partType+" "+radius+" "+rotation+" "+xCent+" "+yCent+
                 " "+zCent+" "+matCode);
      }
      pw.close();
      fw.close();
    } catch (Exception e) {
      System.out.println("Could not write to "+particleFile.getName());
    }
  }

  // Set and get the rve size
  public void setRVESize(double rveSize) {d_rveSize = rveSize;}
  public double getRVESize() {return d_rveSize;}

  // Get particleList data
  public int size() {return d_particleList.size();}

  public Particle getParticle(int index) {
    if (index > d_particleList.size() || index < 0) return null;
    return (Particle) d_particleList.elementAt(index);
  }

  // Set particleList data
  public void addParticle(Particle particle) {
    if (particle != null) {
      d_particleList.addElement(particle);
    }
  }

  // Check is the particle list is empty
  public boolean isEmpty() {
    if (!(d_particleList.size() > 0)) return true;
    return false;
  }

  // Clear particle list
  public void clear() {
    if (!isEmpty()) {
      d_particleList.clear();
      d_triangleList.clear();
      d_voronoiList.clear();
    }
  }

  /**
   * Triangulate the particle list
   */
  public void triangulate() {
    Voronoi vor = new Voronoi(this);
    vor.process();
  }

  /**
   *  Add a triangle to the triangle list
   */
  public void addTriangle(PolygonDouble p) {
    d_triangleList.addElement(p);
  }

  /**
   *  Get the triangle list
   */
  public Vector getTriangles() {
    return d_triangleList;
  }

  /**
   *  Add a point to the voronoi vertex list
   */
  public void addVoronoiVertex(Point p) {
    d_voronoiList.addElement(p);
  }

  /**
   *  Get the voronoi vertex list
   */
  public Vector getVoronoiVertices() {
    return d_voronoiList;
  }

}
