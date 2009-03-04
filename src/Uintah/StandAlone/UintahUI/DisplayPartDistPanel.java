//**************************************************************************
// Program : DisplayPartDistPanel.java
// Purpose : Create a frame that contains widgets to 
//           1) Display the initial distribution of particle sizes.
//           2) Display the distribution to be used for generating 
//              the random spatial locations.
//           3) Display the relative particle sizes.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.text.DecimalFormat;

//**************************************************************************
// Class   : DisplayPartDistPanel
// Purpose : Creates three canvas displays for plotting the data. Reads
//           the calculated data from InputPartDistFrame class and uses
//           the canvases to plot the data.
//**************************************************************************
public class DisplayPartDistPanel extends JPanel {

  private ParticleSize d_partSizeDist = null;
  private ParticleSizeDistInputPanel d_parent = null;

  private DistribCanvas inputCanvas = null;
  private DistribCanvas calcCanvas = null;
  private BallCanvas ballCanvas = null;

  int INPUT = 1;
  int CALC = 2;
  
  public DisplayPartDistPanel(ParticleSize partSizeDist, 
                              ParticleSizeDistInputPanel parent) {

    // Save the input arguments
    d_partSizeDist = partSizeDist;
    d_parent = parent;

    // set the size
    //setLocation(100,100);
    int canvasWidth = 100;
    int canvasHeight = 100;

    // Create the panels to contain the components
    JPanel panel1 = new JPanel(new GridLayout(1,0)); 
    JPanel panel11 = new JPanel();
    JPanel panel12 = new JPanel();
    JPanel panel2 = new JPanel();

    // Set the layouts for the panels that do not have layouts
    GridBagLayout gbInput = new GridBagLayout();
    GridBagLayout gbCalc = new GridBagLayout();
    GridBagLayout gbBall = new GridBagLayout();
    panel11.setLayout(gbInput);
    panel12.setLayout(gbCalc);
    panel2.setLayout(gbBall);

    // Create the components
    JLabel inputCanvasLabel = new JLabel("Input Size Distribution");
    inputCanvas = new DistribCanvas(3*canvasWidth, 3*canvasHeight, INPUT);
    JLabel calcCanvasLabel = new JLabel("Calculated Size Distribution");
    calcCanvas = new DistribCanvas(3*canvasWidth, 3*canvasHeight, CALC);
    JLabel ballCanvasLabel = new JLabel("Calculated Particle Sizes");
    ballCanvas = new BallCanvas(6*canvasWidth, canvasHeight);

    // Add the components to the corresponding panels
    GridBagConstraints gbc = new GridBagConstraints();
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                    1.0,1.0, 0,0, 1,1, 5);
    gbInput.setConstraints(inputCanvasLabel, gbc);
    panel11.add(inputCanvasLabel);
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
                                    1.0,1.0, 0,1, 1,1, 5);
    gbInput.setConstraints(inputCanvas, gbc);
    panel11.add(inputCanvas);
    panel1.add(panel11);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                    1.0,1.0, 0,0, 1,1, 5);
    gbCalc.setConstraints(calcCanvasLabel, gbc);
    panel12.add(calcCanvasLabel);
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
                                    1.0,1.0, 0,1, 1,1, 5);
    gbCalc.setConstraints(calcCanvas, gbc);
    panel12.add(calcCanvas);
    panel1.add(panel12);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                    1.0,1.0, 0,0, 1,1, 5);
    gbBall.setConstraints(ballCanvasLabel, gbc);
    panel2.add(ballCanvasLabel);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                                    1.0,1.0, 0,1, 1,1, 5);
    gbBall.setConstraints(ballCanvas, gbc);
    panel2.add(ballCanvas);

    // Create a gridbag and constraints for the three panels
    GridBagLayout gb = new GridBagLayout();
    setLayout(gb);
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
                                    1.0,1.0, 0,0, 1,1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1);
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
                                    1.0,1.0, 0,1, 1,1, 1);
    gb.setConstraints(panel2, gbc);
    add(panel2);
  }

  // Method for setting the particle distribution
  public void setParticleSizeDist(ParticleSize ps) {
    d_partSizeDist = ps;
  }

  // Refresh the plots
  public void refresh() {
    inputCanvas.refresh();
    calcCanvas.refresh();
    ballCanvas.refresh();
  }

  //
  // Class : DistribCanvas
  // Type  : Inner private
  // Action: Draws a canvas containing the cumulative distribution of the
  //         particles, the scales etc.
  // 
  class DistribCanvas extends LightWeightCanvas {

    // Data
    int xbuf, ybuf, xsmallbuf, ysmallbuf, xmin, ymin, xmax, ymax;
    int xshortTick, yshortTick, xmedTick, ymedTick, xlongTick, ylongTick;
    int d_flag = INPUT;

    // Constructor
    public DistribCanvas(int width, int height, int flag) {

      // set the size of the canvas
      super(width, height);

      // initialize
      initialize();
      d_flag = flag;
    }

    // method to initialize the canvas and draw the data
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

    // method to paint the components
    public void paintComponent(Graphics g) {

      // Draw the histograms and cumulative distribution
      drawHistogram(g);

      // Draw the rules
      drawRule(g);
    }

    // method to refresh the components
    public void refresh() {
      Graphics g = getGraphics();
      clear(g);
      paintComponent(g);
    }

    // method to clear the component of existing stuff
    public void clear(Graphics g) {

      // Draw the gray box
      Dimension d = getSize();
      g.setColor(getBackground());
      g.fillRect(0,0,d.width,d.height);
    }

    // Method to draw the Rule
    private void drawRule(Graphics g) {

      // Get the particle size data to fix the dimensions of the axes
      int nofSizesInp = d_partSizeDist.nofSizesInp;
      if (nofSizesInp == 0) return;
      int nofSizesCalc = d_partSizeDist.nofSizesCalc;

      // Get the maximum particle size and its exponent and mantissa
      double maxPartSize = 1000.0;
      if (d_flag == INPUT) {
        maxPartSize = d_partSizeDist.sizeInp[nofSizesInp-1];
      } else {
        maxPartSize = d_partSizeDist.sizeCalc[nofSizesCalc-1];
      }
      double[] expomanti = computeExponentMantissa(maxPartSize);
      double partSizeExponent = expomanti[0];
      double partSizeMantissa = expomanti[1];
      
      double scale = 100.0;
      int maxSize = Math.round((float)(partSizeMantissa*scale));
      int sizeIncr = maxSize/10;

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
      g.drawString("Particle Size (x 1.0e" + 
                   String.valueOf(Math.round((float)(partSizeExponent-2.0))) +
                   ")", (xmax+xmin)/3,ymax+ybuf);
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

      // Plot the ticks in the y direction
      int yloc = ymax;
      incr = (ymax-ymin)/10;
      if (d_flag != INPUT) 
        g.drawString("N",0,(ymax+ymin)/2);
      else
        g.drawString("Vol %",0,(ymax+ymin)/2-xmedTick);
      for (int i = 0; i <= 10; i++) {
        if (i%10 == 0) {
          g.drawLine(xmin, yloc, xmin-xlongTick, yloc);
          g.drawString(String.valueOf(i*10),2,yloc);
        } else if (i%2 == 0) {
          g.drawLine(xmin, yloc, xmin-xshortTick, yloc);
          g.drawString(String.valueOf(i*10),xmin-xlongTick,yloc);
        } else {
          g.drawLine(xmin, yloc, xmin-xshortTick, yloc);
          g.drawString(String.valueOf(i*10),xmin-xlongTick,yloc);
        }
        yloc -= incr;
      } 
    }

    // Compute exponent and mantissa of a double
    private double[] computeExponentMantissa(double val) {

      // Find the mantissa and exponent of a double
      double exp = Math.abs(Math.log(val)/Math.log(10.0));
      if (val < 1.0) {
        exp = - Math.ceil(exp);
      } else {
        exp = Math.floor(exp);
      }
      double man = val*Math.pow(10.0,-exp);  
      double output[] = new double[2];
      output[0] = exp;
      output[1] = man;
      return output;
    }

    // Draw the cumulative distribution and histograms of the particle
    // sizes
    private void drawHistogram(Graphics g) {

      // Read the data to be used
      int nofSizesInp = d_partSizeDist.nofSizesInp;
      if (nofSizesInp == 0) return;
      int nofSizesCalc = d_partSizeDist.nofSizesCalc;

      // Get the maximum particle size and its exponent and mantissa
      double maxPartSize = 1000.0;
      if (d_flag == INPUT) {
        maxPartSize = d_partSizeDist.sizeInp[nofSizesInp-1];
      } else {
        maxPartSize = d_partSizeDist.sizeCalc[nofSizesCalc-1];
      }
      double[] expomanti = computeExponentMantissa(maxPartSize);
      double partSizeExponent = expomanti[0];
      double partSizeMantissa = expomanti[1];

      double scale = 100.0;
      int maxSize = Math.round((float)(partSizeMantissa*scale));

      if (d_flag == INPUT) {

        // Draw the input vol frac histogram
        double cum1 = 0.0;
        double cum2 = 0.0;
        for (int ii = 0; ii < nofSizesInp; ii++) {

          // Draw the histogram
          double size_start = 0.0;
          if (ii > 0) size_start = d_partSizeDist.sizeInp[ii-1];
          double size_end = d_partSizeDist.sizeInp[ii];

          size_start *= (partSizeMantissa*scale/maxPartSize);
          size_end *= (partSizeMantissa*scale/maxPartSize);

          int minXBox = getXScreenCoord(size_start, maxSize);
          int minYBox = getYScreenCoord(d_partSizeDist.volFracInp[ii]);
          int maxXBox = getXScreenCoord(size_end, maxSize);
          int maxYBox = getYScreenCoord(0.0);
          int boxWidth = maxXBox-minXBox;
          int boxHeight = maxYBox-minYBox;
          
          // Draw the box
          g.setColor(new Color(184,119,27));
          g.fillRect(minXBox, minYBox, boxWidth, boxHeight);
          g.setColor(new Color(0,0,0));
          g.drawRect(minXBox, minYBox, boxWidth, boxHeight);
                                         
          // Draw the cumulative distribution of the input
          int x1 = getXScreenCoord(size_start, maxSize);
          int x2 = getXScreenCoord(size_end, maxSize);
          int y1 = getYScreenCoord(cum1);
          cum1 += d_partSizeDist.volFracInp[ii];
          int y2 = getYScreenCoord(cum1);
          g.setColor(new Color(184,60,27));
          g.drawLine(x1,y1,x2,y2);
          g.drawLine(x1+1,y1,x2+1,y2);
          g.drawLine(x1+2,y1,x2+2,y2);
          g.setColor(new Color(0,0,0));
        }

        // Draw the calculated vol frac histogram
        double cum1Calc = 0.0;
        double cum2Calc = 0.0;
        for (int ii = 0; ii < nofSizesCalc; ii++) {

          // Draw the histogram
          double size_start = 0.0;
          if (ii > 0) size_start = d_partSizeDist.sizeCalc[ii-1];
          double size_end = d_partSizeDist.sizeCalc[ii];

          size_start *= (partSizeMantissa*scale/maxPartSize);
          size_end *= (partSizeMantissa*scale/maxPartSize);

          int minXBox = getXScreenCoord(size_start, maxSize);
          int minYBox = getYScreenCoord(d_partSizeDist.volFrac3DCalc[ii]);
          int maxXBox = getXScreenCoord(size_end, maxSize);
          int maxYBox = getYScreenCoord(0.0);
          int boxWidth = maxXBox-minXBox;
          int boxHeight = maxYBox-minYBox;
          
          // Draw the box
          g.setColor(new Color(200,200,10));
          g.fillRect(minXBox, minYBox, boxWidth, boxHeight);
          g.setColor(new Color(0,0,0));
          g.drawRect(minXBox, minYBox, boxWidth, boxHeight);
          g.setColor(new Color(0,0,0));

          // Draw the cumulative distribution of computed vol frac 
          int x1 = getXScreenCoord(size_start, maxSize);
          int x2 = getXScreenCoord(size_end, maxSize);
          int y1 = getYScreenCoord(cum1Calc);
          cum1Calc += d_partSizeDist.volFrac3DCalc[ii];
          int y2 = getYScreenCoord(cum1Calc);
          g.setColor(new Color(200,200,10));
          g.drawLine(x1,y1,x2,y2);
          g.drawLine(x1+1,y1,x2+1,y2);
          g.drawLine(x1+2,y1,x2+2,y2);
          g.setColor(new Color(0,0,0));
        }

        // Put the labels on the plot
        int x0 = xmin+xbuf;
        int y0 = ymin+yshortTick;
        g.setColor(new Color(184,119,27));
        g.fillRect(x0,y0,xshortTick,yshortTick);
        g.setColor(new Color(0,0,0));
        g.drawRect(x0,y0,xshortTick,yshortTick);
        g.setColor(new Color(184,60,27));
        y0 += yshortTick/2;
        g.drawLine(x0+xmedTick, y0, x0+xlongTick, y0);
        y0 += yshortTick/2;
        g.drawString("Input",x0+xbuf,y0);
        y0 = ymin+ylongTick;
        g.setColor(new Color(200,200,10));
        g.fillRect(x0,y0,xshortTick,yshortTick);
        g.drawRect(x0,y0,xshortTick,yshortTick);
        y0 += yshortTick/2;
        g.drawLine(x0+xmedTick, y0, x0+xlongTick, y0);
        y0 += yshortTick/2;
        g.drawString("Calculated",x0+xbuf,y0);
        g.setColor(new Color(0,0,0));

      } else {
        
        // Put the labels on the plot
        int x0 = (xmax-xmin)/2;
        int y0 = ymin+yshortTick;
        y0 += yshortTick/2;
        g.setColor(new Color(84,27,225));
        g.drawLine(x0+xmedTick, y0, x0+xlongTick, y0);
        y0 += yshortTick/2;
        g.drawString("Distribution in 2D",x0+xbuf,y0);
        y0 = ymin+ylongTick;
        y0 += yshortTick/2;
        g.setColor(new Color(184,119,27));
        g.drawLine(x0+xmedTick, y0, x0+xlongTick, y0);
        y0 += yshortTick/2;
        g.drawString("Distribution in 3D",x0+xbuf,y0);
        g.setColor(new Color(0,0,0));

        // Find the total number of balls
        double numBalls2D = 0.0;
        double numBalls3D = 0.0;
        for (int ii = 0; ii < nofSizesCalc; ii++) {
          numBalls2D += d_partSizeDist.freq2DCalc[ii];
          numBalls3D += d_partSizeDist.freq3DCalc[ii];
        }
        numBalls2D /= 100.0;
        numBalls3D /= 100.0;

        // Draw the lines showing the distribution of balls
        double cum1 = 0.0;
        double cum2 = 0.0;
        for (int ii = 0; ii < nofSizesCalc; ii++) {

          double size_start = 0.0;
          if (ii > 0) size_start = d_partSizeDist.sizeCalc[ii-1];
          double size_end = d_partSizeDist.sizeCalc[ii];
          size_start *= (partSizeMantissa*scale/maxPartSize);
          size_end *= (partSizeMantissa*scale/maxPartSize);

          double freq2D_start = 0.0;
          double freq3D_start = 0.0;
          if (ii > 0) {
            freq2D_start = d_partSizeDist.freq2DCalc[ii-1]/numBalls2D;
            freq3D_start = d_partSizeDist.freq3DCalc[ii-1]/numBalls3D;
          }
          double freq2D_end = d_partSizeDist.freq2DCalc[ii]/numBalls2D;
          double freq3D_end = d_partSizeDist.freq3DCalc[ii]/numBalls3D;

          int x1 = getXScreenCoord(size_start, maxSize);
          int x2 = getXScreenCoord(size_end, maxSize);
          int y1 = getYScreenCoord(freq2D_start);
          int y2 = getYScreenCoord(freq2D_end);
          g.setColor(new Color(84,27,225));
          g.drawLine(x1,y1,x2,y2);
          g.drawLine(x1+1,y1,x2+1,y2);
          g.drawLine(x1+2,y1,x2+2,y2);

          y1 = getYScreenCoord(freq3D_start);
          y2 = getYScreenCoord(freq3D_end);
          g.setColor(new Color(184,119,27));
          g.drawLine(x1,y1,x2,y2);
          g.drawLine(x1+1,y1,x2+1,y2);
          g.drawLine(x1+2,y1,x2+2,y2);

          g.setColor(new Color(0,0,0));

          // Draw the cumulative distribution of the frequencies
          y1 = getYScreenCoord(cum1);
          cum1 += freq2D_end;
          y2 = getYScreenCoord(cum1);
          g.setColor(new Color(84,27,225));
          g.drawLine(x1,y1,x2,y2);
          g.drawLine(x1+1,y1,x2+1,y2);
          g.drawLine(x1+2,y1,x2+2,y2);

          y1 = getYScreenCoord(cum2);
          cum2 += freq3D_end;
          y2 = getYScreenCoord(cum2);
          g.setColor(new Color(184,119,27));
          g.drawLine(x1,y1,x2,y2);
          g.drawLine(x1+1,y1,x2+1,y2);
          g.drawLine(x1+2,y1,x2+2,y2);
          g.setColor(new Color(0,0,0));
        }
      }
    }

    // Get the screen co=ordinates of a world point
    private int getXScreenCoord(double coord, int maxSize) {
      return xmin+(int) (coord/(double)maxSize*(double)(xmax-xmin));
    }
    private int getYScreenCoord(double coord) {
      return ymax-(int) (coord/100.0*(double)(ymax-ymin));
    }
  }

  //
  // Class : BallCanvas
  // Type  : Inner private
  // Action: Draws a canvas that shows the ball sizes
  //
  class BallCanvas extends LightWeightCanvas {

    // Data

    // Constructor
    public BallCanvas(int width, int height) {

      // set the size of the canvas
      super(width, height);

      // set the background color
      setBackground(new Color(255,255,255));
    }

    // method to paint the components
    public void paintComponent(Graphics g) {
      drawBalls(g);
    }

    // method to refresh the components
    public void refresh() {
      Graphics g = getGraphics();
      clear(g);
      paintComponent(g);
    }

    // method to clear the component of existing stuff
    public void clear(Graphics g) {

      // Draw the gray box
      Dimension d = getSize();
      g.setColor(getBackground());
      g.fillRect(0,0,d.width,d.height);
    }

    // method to draw the balls
    private void drawBalls(Graphics g) {

      // get the dimensions of the window
      Dimension d = getSize();
      int width = d.width;
      int height = d.height;

      // Draw a box around the balls
      g.setColor(new Color(255,255,255));
      g.fillRect(2,2,width,height);
      g.setColor(new Color(0,0,0));

      // Read the number of ball sizes
      int nofBallSizes = d_partSizeDist.nofSizesCalc;
      if (nofBallSizes == 0) return;

      // Assign one box to each ball size .. divide the width likewise
      // and a square box in which to place the balls in each box
      int boxWidth = width/nofBallSizes;
      int sqrBoxWidth = boxWidth-5;
      if (boxWidth > height) sqrBoxWidth = height-5;

      // Read the largest ball diameter and assign the scaling factor so
      // that it fits into the square box
      double maxDia = d_partSizeDist.sizeCalc[nofBallSizes-1];
      double scaleFactor = 1/maxDia;

      // calculate the scaled radii of the balls in pixels and the 
      // locations of the centers
      DecimalFormat df = new DecimalFormat("###.#");
      int[] scaledRad = new int[nofBallSizes];
      int[] xCent = new int[nofBallSizes];
      int[] yCent = new int[nofBallSizes];
      for (int ii = 0; ii < nofBallSizes; ii++) {
        scaledRad[ii] = 
          (int)(d_partSizeDist.sizeCalc[ii]*scaleFactor*
                (double)sqrBoxWidth/2.0);
        xCent[ii] = boxWidth/2 + ii*boxWidth;
        yCent[ii] = height/2;
        
        // Draw the balls and the sizes at the center
        g.setColor(new Color(184,119,27));
        g.fillOval(xCent[ii]-scaledRad[ii],yCent[ii]-scaledRad[ii],
                   2*scaledRad[ii],2*scaledRad[ii]);
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g.setColor(new Color(0,0,0));
        String output = df.format(d_partSizeDist.sizeCalc[ii]);
        g.drawString(output, xCent[ii], yCent[ii]);
      }
    }
  }
}
