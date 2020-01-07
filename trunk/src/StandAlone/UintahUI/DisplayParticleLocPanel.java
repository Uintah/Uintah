//**************************************************************************
// Program : DisplayParticleLocPanel.java
// Purpose : Display particle locations.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.util.Random;
import java.util.Vector;
import java.io.*;
import javax.swing.*;

//**************************************************************************
// Class   : DisplayParticleLocPanel
// Purpose : Display particle locations.
//**************************************************************************
public class DisplayParticleLocPanel extends JPanel {

  // Essential data
  private boolean d_isHollow = false;
  private double d_thickness = 0.0;;
  private ParticleList d_partList = null;
  private ParticleLocGeneratePanel d_parent = null;
  
  private TopCanvas topCanvas = null;
  private SideCanvas sideCanvas = null;
  private FrontCanvas frontCanvas = null;

  // static data
  public static final int TOP = 1;
  public static final int SIDE = 2;
  public static final int FRONT = 3;

  public static final int CIRCLE = 1;
  public static final int SPHERE = 2;

  public static final int YES = 1;
  public static final int NO = 2;
  
  public DisplayParticleLocPanel(ParticleList partList,
                                 ParticleLocGeneratePanel parent) {

    // Save the input arguments
    d_isHollow = false;
    d_thickness = 0.0;;
    d_partList = partList;
    d_parent = parent;

    // Create a panel to contain the components
    JPanel panel = new JPanel(new GridLayout(2,2));

    // Create the components
    topCanvas = new TopCanvas(300,300);
    sideCanvas = new SideCanvas(300,300);
    frontCanvas = new FrontCanvas(300,300);

    // Add the components to the panels
    panel.add(topCanvas);
    panel.add(sideCanvas);
    panel.add(frontCanvas);

    // Create a gridbag and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                                    1.0,1.0, 0,2, 1,1, 5);
    gb.setConstraints(panel, gbc);
    add(panel);

  }

  //-------------------------------------------------------------------------
  // Refresh the plots
  //-------------------------------------------------------------------------
  public void refresh() {
    topCanvas.refresh();
    sideCanvas.refresh();
    frontCanvas.refresh();
  }

  //**************************************************************************
  // Class   : PlaneCanvas
  // Purpose : Draws a LightWeightCanvas for displaying the planes.
  //           More detailed canvases ar derived from this.
  //**************************************************************************
  protected class PlaneCanvas extends LightWeightCanvas {

    // Data
    int d_type = 0;
    protected int xbuf, ybuf, xsmallbuf, ysmallbuf, xmin, ymin, xmax, ymax;
    protected int xshortTick, yshortTick, xmedTick, ymedTick, xlongTick, 
                  ylongTick;
    protected double d_rveSize;

    //-------------------------------------------------------------------------
    // Constructor
    //-------------------------------------------------------------------------
    public PlaneCanvas(int width, int height, int type) {
      super(width,height);
      d_type = type;
      d_rveSize = d_parent.getRVESize();
      initialize();
    }

    //-------------------------------------------------------------------------
    // initialize
    //-------------------------------------------------------------------------
    private void initialize() {

      // Calculate the buffer area of the canvas
      Dimension d = getSize();
      xbuf = d.width/10;
      ybuf = d.height/10;
      xsmallbuf = xbuf/4;
      ysmallbuf = ybuf/4;

      // Calculate the drawing limits
      xmin = xbuf;
      ymin = ybuf;
      xmax = d.width-xbuf;
      ymax = d.height-ybuf;

      // Calculate the tick lengths
      xlongTick = xsmallbuf*3;
      xmedTick = xsmallbuf*2;
      xshortTick = xsmallbuf;
      ylongTick = ysmallbuf*3;
      ymedTick = ysmallbuf*2;
      yshortTick = ysmallbuf;
    }

    //-------------------------------------------------------------------------
    // paint components
    //-------------------------------------------------------------------------
    public void paintComponent(Graphics g) {

      // Draw the rules
      d_rveSize = d_parent.getRVESize();
      drawRule(g);
    }

    //-------------------------------------------------------------------------
    // paint the component immediately
    //-------------------------------------------------------------------------
    public void paintImmediately() {
      paintImmediately(xmin, xmax, xmax-xmin, ymax-ymin);
    }
    public void paintImmediately(int x, int y, int w, int h) {
      Graphics g = getGraphics();
      super.paintImmediately(x, y, w, h);
      d_rveSize = d_parent.getRVESize();
      drawRule(g);
    }

    //-------------------------------------------------------------------------
    // Method to draw the Rule
    //-------------------------------------------------------------------------
    private void drawRule(Graphics g) {

      // Get the particle size data to fix the dimensions of the axes
      double sizeIncr = d_rveSize/10.0;

      // Draw the highlighted rects
      g.setColor(new Color(230,163,4));
      g.fillRect(xmin-xbuf,ymin,xbuf,ymax-ymin);
      g.fillRect(xmin,ymax,xmax-xmin,ybuf);
      g.setColor(new Color(0,0,0));

      // Draw the box
      g.drawRect(xmin,ymin,xmax-xmin,ymax-ymin);

      // Plot the ticks in the x direction
      g.setFont(new Font("SansSerif", Font.PLAIN, 10));
      int xloc = xmin;
      int incr = (xmax-xmin)/10;
      String viewType = null;
      String xAxis = null;
      String yAxis = null;
      if (d_type == TOP) {
        viewType = "Top View";
        xAxis = "X, 1";
        yAxis = "Y, 2";
      }
      else if (d_type == SIDE) {
        viewType = "Side View";
        xAxis = "Y, 2";
        yAxis = "Z, 3";
      }
      else if (d_type == FRONT) {
        viewType = "Front View";
        xAxis = "X, 1";
        yAxis = "Z, 3";
      }
      g.drawString(viewType,(xmax+xmin)/3,ymax+ybuf);
      for (int i = 0; i <= 10; i++) {
        if (i%10 == 0) {
          g.drawLine(xloc, ymax, xloc, ymax+ylongTick);
          g.drawString(String.valueOf(i*sizeIncr),xloc-xshortTick,ymax+ybuf-2);
        } else if (i%2 == 0) {
          g.drawLine(xloc, ymax, xloc, ymax+yshortTick);
          g.drawString(String.valueOf(i*sizeIncr),xloc-xshortTick,
                       ymax+ymedTick+2);
        } else {
          g.drawLine(xloc, ymax, xloc, ymax+yshortTick);
          g.drawString(String.valueOf(i*sizeIncr),xloc-xshortTick,
                       ymax+ymedTick+2);
        }
        xloc += incr;
      } 
      g.drawString(xAxis, xmax+xshortTick, ymax);

      // Plot the ticks in the y direction
      int yloc = ymax;
      incr = (ymax-ymin)/10;
      for (int i = 0; i <= 10; i++) {
        if (i%10 == 0) {
          g.drawLine(xmin, yloc, xmin-xlongTick, yloc);
          g.drawString(String.valueOf(i*sizeIncr),2,yloc);
        } else if (i%2 == 0) {
          g.drawLine(xmin, yloc, xmin-xshortTick, yloc);
          g.drawString(String.valueOf(i*sizeIncr),xmin-xlongTick,yloc);
        } else {
          g.drawLine(xmin, yloc, xmin-xshortTick, yloc);
          g.drawString(String.valueOf(i*sizeIncr),xmin-xlongTick,yloc);
        }
        yloc -= incr;
      } 
      g.drawString(yAxis, xmin, ymin-yshortTick);
    }

    //-------------------------------------------------------------------------
    // Get the screen co=ordinates of a world point
    //-------------------------------------------------------------------------
    protected int getXScreenCoord(double coord) {
      return xmin+(int) (coord/d_rveSize*(double)(xmax-xmin));
    }
    protected int getYScreenCoord(double coord) {
      return ymax-(int) (coord/d_rveSize*(double)(ymax-ymin));
    }
    protected int getXScreenLength(double length) {
      return (int) (length/d_rveSize*(double)(xmax-xmin));
    }
    protected int getYScreenLength(double length) {
      return (int) (length/d_rveSize*(double)(ymax-ymin));
    }
  }

  //**************************************************************************
  // Class   : TopCanvas
  // Purpose : Draws a LightWeightCanvas for displaying the planes.
  //           More detailed canvases ar derived from this.
  //**************************************************************************
  private final class TopCanvas extends PlaneCanvas {

    // Data

    //-------------------------------------------------------------------------
    // Constructor
    //-------------------------------------------------------------------------
    public TopCanvas(int width, int height) {
      super(width,height,TOP);
      initialize();
    }

    //-------------------------------------------------------------------------
    // initialize
    //-------------------------------------------------------------------------
    private void initialize() {
    }

    //-------------------------------------------------------------------------
    // paint components
    //-------------------------------------------------------------------------
    public void paintComponent(Graphics g) {
      super.paintComponent(g);
      drawParticles(g);
    }

    //-------------------------------------------------------------------------
    // paint the component immediately
    //-------------------------------------------------------------------------
    public void paintImmediately() {
      paintImmediately(xmin, xmax, xmax-xmin, ymax-ymin);
    }
    public void paintImmediately(int x, int y, int w, int h) {
      Graphics g = getGraphics();
      super.paintImmediately(x, y, w, h);
      drawParticles(g);
    }
    
    //-------------------------------------------------------------------------
    // method to refresh the components
    //-------------------------------------------------------------------------
    public void refresh() {
      Graphics g = getGraphics();
      clear(g);
      paintComponent(g);
    }

    //-------------------------------------------------------------------------
    // method to clear the component of existing stuff
    //-------------------------------------------------------------------------
    public void clear(Graphics g) {

      // Draw the gray box
      Dimension d = getSize();
      g.setColor(getBackground());
      g.fillRect(0,0,d.width,d.height);
    }

    //-------------------------------------------------------------------------
    // Draw the particles
    //-------------------------------------------------------------------------
    private void drawParticles(Graphics g) {
      int size = d_partList.size();
      if (!(size > 0)) return;

      // Find particle type
      Particle part = (Particle) d_partList.getParticle(0);
      int type = part.getType();

      // Find whether the particle is hollow
      d_thickness = part.getThickness();
      if (d_thickness > 0.0) d_isHollow = true;

      // Draw the particles
      if (type == CIRCLE) 
        drawCircles(g, size);
      else if (type == SPHERE)
        drawSpheres(g, size);

      // Draw the box
      g.drawRect(xmin,ymin,xmax-xmin,ymax-ymin);
    }

    //-------------------------------------------------------------------------
    // Draw the circles
    //-------------------------------------------------------------------------
    private void drawCircles(Graphics g, int size) {

      for (int ii = 0; ii < size; ii++) {

        // Get the particle data
        Particle part = (Particle) d_partList.getParticle(ii);
        double radius = part.getRadius();
        Point center = part.getCenter();
        double xCent = center.getX();
        double yCent = center.getY();

        if (circleIsOutsideRVE(radius, xCent, yCent)) {
          continue;
        }

        // Draw the circles
        int radXScreen = getXScreenLength(radius);
        int radYScreen = getYScreenLength(radius);
        int xCentScreen = getXScreenCoord(xCent);
        int yCentScreen = getYScreenCoord(yCent);

        // Set the clipping rectangle
        Rectangle clipRect = new Rectangle(xmin, ymin, xmax-xmin, ymax-ymin);
        g.setClip(clipRect);
        g.setColor(new Color(184,119,27));
        g.fillOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
        g.setColor(new Color(0,0,0));
        g.drawOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);

        // If the circles are hollow then plot the inside in 
        // background color
        if (d_isHollow) {
          radXScreen = getXScreenLength(radius-d_thickness);
          radYScreen = getYScreenLength(radius-d_thickness);
          g.setColor(getBackground());
          g.fillOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                     2*radXScreen, 2*radYScreen);
          g.setColor(new Color(0,0,0));
          g.drawOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                     2*radXScreen, 2*radYScreen);
        }
      }
    }

    //-------------------------------------------------------------------------
    // Check if the ball is outside the RVE
    //-------------------------------------------------------------------------
    private boolean circleIsOutsideRVE(double rad, double xCent, double yCent) {
      double distXPlus = d_rveSize - (xCent-rad);
      double distYPlus = d_rveSize - (yCent-rad);
      double distXMinus = xCent+rad;
      double distYMinus = yCent+rad;
      if (distXPlus <= 0.0) return true;
      if (distYPlus <= 0.0) return true;
      if (distXMinus <= 0.0) return true;
      if (distYMinus <= 0.0) return true;
      return false;
    }

    //-------------------------------------------------------------------------
    // Check if the ball is completely inside the RVE
    //-------------------------------------------------------------------------
    private boolean circleIsInsideRVE(double rad, double xCent, double yCent) {
      double distXPlus = d_rveSize - (xCent+rad);
      double distYPlus = d_rveSize - (yCent+rad);
      double distXMinus = xCent-rad;
      double distYMinus = yCent-rad;
      System.out.println("distXPlus = "+distXPlus+" distXMinus="+distXMinus+
                         "distYPlus = "+distXPlus+" distYMinus="+distXMinus);
      if (distXPlus >= 0.0 && distYPlus >= 0 && distXMinus >= 0.0 &&
          distYMinus >= 0.0) return true;
      return false;
    }

    //-------------------------------------------------------------------------
    // If the ball intersects RVE box find arc start angle and arc angle
    //-------------------------------------------------------------------------
    private int[] calcArcAngle(double rad, double xCent, double yCent) {
      double distXPlus = d_rveSize - xCent;
      double distYPlus = d_rveSize - yCent;
      double distXMinus = xCent;
      double distYMinus = yCent;
      int angle1 = 0;
      int angle2 = 0;
      if (distXPlus < rad ) {
        if (distYPlus < rad ) {
          angle1 = 180;
          angle2 = 270;
        } else if (distYMinus < rad ) {
          angle1 = 90;
          angle2 = 180;
        } else {
          angle1 = 90;
          angle2 = 270;
        }
      } else if (distXMinus < rad ) {
        if (distYPlus < rad ) {
          angle1 = 270;
          angle2 = 0;
        } else if (distYMinus < rad ) {
          angle1 = 0;
          angle2 = 90;
        } else {
          angle1 = 270;
          angle2 = 90;
        }
      }
      int[] angle = new int[2];
      angle[0] = angle1;
      angle[1] = angle2 - angle1;
 
      return angle;
    }

    //-------------------------------------------------------------------------
    // Draw the spheres
    //-------------------------------------------------------------------------
    private void drawSpheres(Graphics g, int size) {

      // Store the particle data
      double[] radius = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
        Particle part = (Particle) d_partList.getParticle(ii);
        Point center = part.getCenter();
        radius[ii] = part.getRadius();
        xCent[ii] = center.getX();
        yCent[ii] = center.getY();
        zCent[ii] = center.getZ();
      }

      // sort the particle data in order of ascending Z Coord
      for (int jj = 1; jj < size; jj++) {
        double keyXCent = xCent[jj];
        double keyYCent = yCent[jj];
        double keyZCent = zCent[jj];
        double keyRad = radius[jj];
        int ii = jj-1;
        while (ii >= 0 && zCent[ii] > keyZCent) {
          xCent[ii+1] = xCent[ii];
          yCent[ii+1] = yCent[ii];
          zCent[ii+1] = zCent[ii];
          radius[ii+1] = radius[ii];
          ii--;
        }
        xCent[ii+1] = keyXCent;
        yCent[ii+1] = keyYCent;
        zCent[ii+1] = keyZCent;
        radius[ii+1] = keyRad;
      }

      // Draw the circles 
      for (int ii = 0; ii < size; ii++) {
        int radXScreen = getXScreenLength(radius[ii]);
        int radYScreen = getYScreenLength(radius[ii]);
        int xCentScreen = getXScreenCoord(xCent[ii]);
        int yCentScreen = getYScreenCoord(yCent[ii]);
        g.setColor(new Color(184,119,27));
        g.fillOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
        g.setColor(new Color(0,0,0));
        g.drawOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
      }
    }

  }

  //**************************************************************************
  // Class   : SideCanvas
  // Purpose : Draws a LightWeightCanvas for displaying the planes.
  //           More detailed canvases ar derived from this.
  //**************************************************************************
  private final class SideCanvas extends PlaneCanvas {

    // Data

    //-------------------------------------------------------------------------
    // Constructor
    //-------------------------------------------------------------------------
    public SideCanvas(int width, int height) {
      super(width,height,SIDE);
      initialize();
    }

    //-------------------------------------------------------------------------
    // initialize
    //-------------------------------------------------------------------------
    private void initialize() {
    }

    //-------------------------------------------------------------------------
    // paint components
    //-------------------------------------------------------------------------
    public void paintComponent(Graphics g) {
      super.paintComponent(g);
      drawParticles(g);
    }

    //-------------------------------------------------------------------------
    // method to refresh the components
    //-------------------------------------------------------------------------
    public void refresh() {
      Graphics g = getGraphics();
      clear(g);
      paintComponent(g);
    }

    //-------------------------------------------------------------------------
    // method to clear the component of existing stuff
    //-------------------------------------------------------------------------
    public void clear(Graphics g) {

      // Draw the gray box
      Dimension d = getSize();
      g.setColor(getBackground());
      g.fillRect(0,0,d.width,d.height);
    }

    //-------------------------------------------------------------------------
    // Draw the particles
    //-------------------------------------------------------------------------
    private void drawParticles(Graphics g) {
      int size = d_partList.size();
      if (!(size > 0)) return;

      // Find particle type
      Particle part = (Particle) d_partList.getParticle(0);
      int type = part.getType();

      // Find whether the particle is hollow
      d_thickness = part.getThickness();
      if (d_thickness > 0.0) d_isHollow = true;

      // Draw the particles
      if (type == CIRCLE) 
        drawCylinders(g, size);
      else if (type == SPHERE)
        drawSpheres(g, size);
    }

    //-------------------------------------------------------------------------
    // Draw the cylinders corresponding to the circles with min X
    // cylinders first
    //-------------------------------------------------------------------------
    private void drawCylinders(Graphics g, int size) {

      // Store the particle data
      double[] radius = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
        Particle part = (Particle) d_partList.getParticle(ii);
        Point center = part.getCenter();
        radius[ii] = part.getRadius();
        xCent[ii] = center.getX();
        yCent[ii] = center.getY();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
        double keyXCent = xCent[jj];
        double keyYCent = yCent[jj];
        double keyRad = radius[jj];
        int ii = jj-1;
        while (ii >= 0 && xCent[ii] > keyXCent) {
          xCent[ii+1] = xCent[ii];
          yCent[ii+1] = yCent[ii];
          radius[ii+1] = radius[ii];
          ii--;
        }
        xCent[ii+1] = keyXCent;
        yCent[ii+1] = keyYCent;
        radius[ii+1] = keyRad;
      }

      // Draw the lines next
      int blue = 216;
      for (int ii = 0; ii < size; ii++) {
        int radScreen = getYScreenLength(radius[ii]);
        int centScreen = getYScreenCoord(yCent[ii]);
        int quo = ii%8;
        if (quo >= 7) blue = 27;
        else if (quo == 6) blue = 54;
        else if (quo == 5) blue = 81;
        else if (quo == 4) blue = 108;
        else if (quo == 3) blue = 135;
        else if (quo == 2) blue = 162;
        else if (quo == 1) blue = 189;
        Rectangle clipRect = new Rectangle(xmin, ymin, xmax-xmin, ymax-ymin);
        g.setClip(clipRect);
        g.setColor(new Color(184,119,blue));
        g.fillRect(xmin, centScreen-radScreen, xmax-xmin,2*radScreen);
        g.setColor(new Color(0,0,0));
        g.drawRect(xmin, centScreen-radScreen, xmax-xmin,2*radScreen);
      }
    }

    //-------------------------------------------------------------------------
    // Draw the spheres
    //-------------------------------------------------------------------------
    private void drawSpheres(Graphics g, int size) {

      // Store the particle data
      double[] radius = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
        Particle part = (Particle) d_partList.getParticle(ii);
        Point center = part.getCenter();
        radius[ii] = part.getRadius();
        xCent[ii] = center.getX();
        yCent[ii] = center.getY();
        zCent[ii] = center.getZ();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
        double keyXCent = xCent[jj];
        double keyYCent = yCent[jj];
        double keyZCent = zCent[jj];
        double keyRad = radius[jj];
        int ii = jj-1;
        while (ii >= 0 && xCent[ii] > keyXCent) {
          xCent[ii+1] = xCent[ii];
          yCent[ii+1] = yCent[ii];
          zCent[ii+1] = zCent[ii];
          radius[ii+1] = radius[ii];
          ii--;
        }
        xCent[ii+1] = keyXCent;
        yCent[ii+1] = keyYCent;
        zCent[ii+1] = keyZCent;
        radius[ii+1] = keyRad;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
        int radXScreen = getXScreenLength(radius[ii]);
        int radYScreen = getYScreenLength(radius[ii]);
        int xCentScreen = getXScreenCoord(yCent[ii]);
        int yCentScreen = getYScreenCoord(zCent[ii]);
        g.setColor(new Color(184,119,27));
        g.fillOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
        g.setColor(new Color(0,0,0));
        g.drawOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
      }
    }

  }

  //**************************************************************************
  // Class   : FrontCanvas
  // Purpose : Draws a LightWeightCanvas for displaying the planes.
  //           More detailed canvases ar derived from this.
  //**************************************************************************
  private final class FrontCanvas extends PlaneCanvas {

    // Data

    //-------------------------------------------------------------------------
    // Constructor
    //-------------------------------------------------------------------------
    public FrontCanvas(int width, int height) {
      super(width,height,DisplayParticleLocPanel.FRONT);
      initialize();
    }

    //-------------------------------------------------------------------------
    // initialize
    //-------------------------------------------------------------------------
    private void initialize() {
    }

    //-------------------------------------------------------------------------
    // paint components
    //-------------------------------------------------------------------------
    public void paintComponent(Graphics g) {
      super.paintComponent(g);
      drawParticles(g);
    }

    //-------------------------------------------------------------------------
    // method to refresh the components
    //-------------------------------------------------------------------------
    public void refresh() {
      Graphics g = getGraphics();
      clear(g);
      paintComponent(g);
    }

    //-------------------------------------------------------------------------
    // method to clear the component of existing stuff
    //-------------------------------------------------------------------------
    public void clear(Graphics g) {

      // Draw the gray box
      Dimension d = getSize();
      g.setColor(getBackground());
      g.fillRect(0,0,d.width,d.height);
    }

    //-------------------------------------------------------------------------
    // Draw the particles
    //-------------------------------------------------------------------------
    private void drawParticles(Graphics g) {
      int size = d_partList.size();
      if (!(size > 0)) return;

      // Find particle type
      Particle part = (Particle) d_partList.getParticle(0);
      int type = part.getType();

      // Find whether the particle is hollow
      d_thickness = part.getThickness();
      if (d_thickness > 0.0) d_isHollow = true;

      // Draw the particles
      if (type == CIRCLE) 
        drawCylinders(g, size);
      else if (type == SPHERE)
        drawSpheres(g, size);
    }

    //-------------------------------------------------------------------------
    // Draw the cylinders corresponding to the circles with max Y
    // cylinders first
    //-------------------------------------------------------------------------
    private void drawCylinders(Graphics g, int size) {

      // Store the particle data
      double[] radius = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
        Particle part = (Particle) d_partList.getParticle(ii);
        Point center = part.getCenter();
        radius[ii] = part.getRadius();
        xCent[ii] = center.getX();
        yCent[ii] = center.getY();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
        double keyXCent = xCent[jj];
        double keyYCent = yCent[jj];
        double keyRad = radius[jj];
        int ii = jj-1;
        while (ii >= 0 && yCent[ii] < keyYCent) {
          xCent[ii+1] = xCent[ii];
          yCent[ii+1] = yCent[ii];
          radius[ii+1] = radius[ii];
          ii--;
        }
        xCent[ii+1] = keyXCent;
        yCent[ii+1] = keyYCent;
        radius[ii+1] = keyRad;
      }

      // Draw the lines next
      int blue = 216;
      for (int ii = 0; ii < size; ii++) {
        int radScreen = getXScreenLength(radius[ii]);
        int centScreen = getXScreenCoord(xCent[ii]);
        int quo = ii%8;
        if (quo >= 7) blue = 27;
        else if (quo == 6) blue = 54;
        else if (quo == 5) blue = 81;
        else if (quo == 4) blue = 108;
        else if (quo == 3) blue = 135;
        else if (quo == 2) blue = 162;
        else if (quo == 1) blue = 189;
        Rectangle clipRect = new Rectangle(xmin, ymin, xmax-xmin, ymax-ymin);
        g.setClip(clipRect);
        g.setColor(new Color(184,119,blue));
        g.fillRect(centScreen-radScreen,ymin,2*radScreen,ymax-ymin);
        g.setColor(new Color(0,0,0));
        g.drawRect(centScreen-radScreen,ymin,2*radScreen,ymax-ymin);
      }
    }

    //-------------------------------------------------------------------------
    // Draw the spheres
    //-------------------------------------------------------------------------
    private void drawSpheres(Graphics g, int size) {

      // Store the particle data
      double[] radius = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
        Particle part = (Particle) d_partList.getParticle(ii);
        Point center = part.getCenter();
        radius[ii] = part.getRadius();
        xCent[ii] = center.getX();
        yCent[ii] = center.getY();
        zCent[ii] = center.getZ();
      }

      // sort the particle data in order of descending Y Coord
      for (int jj = 1; jj < size; jj++) {
        double keyXCent = xCent[jj];
        double keyYCent = yCent[jj];
        double keyZCent = zCent[jj];
        double keyRad = radius[jj];
        int ii = jj-1;
        while (ii >= 0 && yCent[ii] < keyYCent) {
          xCent[ii+1] = xCent[ii];
          yCent[ii+1] = yCent[ii];
          zCent[ii+1] = zCent[ii];
          radius[ii+1] = radius[ii];
          ii--;
        }
        xCent[ii+1] = keyXCent;
        yCent[ii+1] = keyYCent;
        zCent[ii+1] = keyZCent;
        radius[ii+1] = keyRad;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
        int radXScreen = getXScreenLength(radius[ii]);
        int radYScreen = getYScreenLength(radius[ii]);
        int xCentScreen = getXScreenCoord(xCent[ii]);
        int yCentScreen = getYScreenCoord(zCent[ii]);
        g.setColor(new Color(184,119,27));
        g.fillOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
        g.setColor(new Color(0,0,0));
        g.drawOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
                   2*radXScreen, 2*radYScreen);
      }
    }

  }
}
