/**
 * Class   : PolygonDouble
 * Purpose : Create a polygon with co-oordinates that have double values
 * Author  : Biswajit
 */

public class PolygonDouble {
  
  /**
   * Private data
   */
  double[] d_x;
  double[] d_y;
  int d_nPts;

  /**
   * Constructor
   */
  PolygonDouble() {
    d_nPts = 0;
    d_x = new double[20];
    d_y = new double[20];
  }

  /**
   * Constructor
   */
  PolygonDouble(double[] x, double[] y, int nPts) {
    d_nPts = nPts;
    d_x = new double[nPts];
    d_y = new double[nPts];
    for (int ii = 0; ii < nPts; ii++) {
      d_x[ii] = x[ii];
      d_y[ii] = y[ii];
    }
  }

  /**
   * Get the number of points in the polygon
   */
  int nofPoints() {
    return d_nPts;
  }

  /**
   * Get a point at an index
   */
  double x(int index) { return d_x[index]; }
  double y(int index) { return d_y[index]; }

  /**
   * Add a point to the polygon
   */
  void add(double x, double y) {
    d_x[d_nPts] = x;
    d_y[d_nPts] = y;
    ++d_nPts;
  }

  /**
   * Print the polygon
   */
  void print() {
    System.out.println("NofPts = "+d_nPts);
    for (int ii = 0; ii < d_nPts; ii++) {
      System.out.println("x = "+d_x[ii]+" y = "+d_y[ii]);
    }
  }
}
