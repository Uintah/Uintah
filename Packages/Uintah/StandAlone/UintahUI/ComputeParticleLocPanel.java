//**************************************************************************
// Program : ComputeParticleLocPanel.java
// Purpose : Create a panel that contains textFields to take inputs
//           and buttons that trigger events to calculate the location
//           of the particles in the given size distribution in a box
//           in a random manner.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.util.Random;
import java.io.*;
import javax.swing.*;

//**************************************************************************
// Class   : ComputeParticleLocPanel
// Purpose : Generate the locations of randomly distributed particles of a
//           particular size distribution.
//**************************************************************************
public class ComputeParticleLocPanel extends JPanel 
                                     implements ItemListener,
                                                ActionListener {

  // Essential data
  private ParticleList d_partList = null;
  private ParticleSize d_partSizeDist = null;
  private ParticleLocGeneratePanel d_parent = null;
  
  // Local RVE particle size distribution
  private ParticleSize d_rvePartSizeDist = null;
  private int d_partTypeFlag = 0;
  private boolean d_hollowFlag = false;
  private double d_thickness = 0.0;

  // Data that are stored for the life of the object
  private DecimalField rveSizeEntry = null;
  private JLabel thicknessLabel = null;
  private DecimalField thicknessEntry = null;
  private JComboBox partTypeCBox = null;
  private JButton randomButton = null;
  private JButton periodicButton = null;
  private JButton saveButton = null;

  // Other data
  private double d_rveSize = 0.0;

  // static data
  public static final int CIRCLE = 1;
  public static final int SPHERE = 2;

  public static final int YES = 1;
  public static final int NO = 2;
  
  public ComputeParticleLocPanel(ParticleList partList,
                                 ParticleSize partSizeDist,
                                 ParticleLocGeneratePanel parent) {

    // Save the input arguments
    d_partList = partList;
    d_partSizeDist = partSizeDist;
    d_parent = parent;

    // Initialize local RVE particle size distribution
    d_rvePartSizeDist = new ParticleSize(partSizeDist);

    // Initialize flags
    d_partTypeFlag = CIRCLE;
    d_hollowFlag = false;
    d_thickness = 0.0;
    d_rveSize = 100.0;
    d_parent.setRVESize(d_rveSize);

    // Create a gridbag and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the first panel and the components
    JPanel panel1 = new JPanel(new GridLayout(4,0));
    JLabel rveSizeLabel = new JLabel("RVE Size (in one dimension) ");
    panel1.add(rveSizeLabel);


    rveSizeEntry = new DecimalField(1.0,5);
    panel1.add(rveSizeEntry);

    JLabel partTypeLabel = new JLabel("Type of particle ");
    panel1.add(partTypeLabel);

    partTypeCBox = new JComboBox();
    partTypeCBox.addItem("Solid Circle");
    partTypeCBox.addItem("Hollow Circle");
    partTypeCBox.addItem("Solid Sphere");
    partTypeCBox.addItem("Hollow Sphere");
    partTypeCBox.setSelectedIndex(0);
    panel1.add(partTypeCBox);

    thicknessLabel = new JLabel("Thickness");
    panel1.add(thicknessLabel);

    thicknessEntry = new DecimalField(0.0, 9, true);
    panel1.add(thicknessEntry);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,0, 1,1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1);

    // Create the second panel and the components
    JPanel panel10 = new JPanel(new GridLayout(4,0));

    JLabel labelrun = new JLabel("Click on one of the following buttons");
    panel10.add(labelrun);

    randomButton = new JButton("Create Random Distribution");
    randomButton.setActionCommand("dist");
    panel10.add(randomButton);

    periodicButton = new JButton("Create Periodic Distribution");
    periodicButton.setActionCommand("periodic");
    panel10.add(periodicButton);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,1, 1,1, 5);
    gb.setConstraints(panel10, gbc);
    add(panel10);

    // Create the third panel and the components
    JPanel panel3 = new JPanel(new GridLayout(1,0));

    saveButton = new JButton("Save Calculated Data");
    saveButton.setActionCommand("save");
    panel3.add(saveButton);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                                    1.0,1.0, 0,3, 1,1, 5);
    gb.setConstraints(panel3, gbc);
    add(panel3);

    // Create and add the listeners
    partTypeCBox.addItemListener(this);
    randomButton.addActionListener(this);
    periodicButton.addActionListener(this);
    saveButton.addActionListener(this);

    // Disable the thickness label by default
    thicknessLabel.setEnabled(false);
    thicknessEntry.setEnabled(false);
  }

  //--------------------------------------------------------------------------
  // Determine which combo box item was chosen
  //--------------------------------------------------------------------------
  public void itemStateChanged(ItemEvent e) {

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());
    if (item.equals("Solid Circle")) {
      d_partTypeFlag = CIRCLE;
      d_hollowFlag = false;
    } else if (item.equals("Hollow Circle")) {
      d_partTypeFlag = CIRCLE;
      d_hollowFlag = true;
    } else if (item.equals("Solid Sphere")) {
      d_partTypeFlag = SPHERE;
      d_hollowFlag = false;
    } else if (item.equals("Hollow Sphere")) {
      d_partTypeFlag = SPHERE;
      d_hollowFlag = true;
    } 
    thicknessLabel.setEnabled(d_hollowFlag);
    thicknessEntry.setEnabled(d_hollowFlag);
  }

  //--------------------------------------------------------------------------
  // Actions performed after a button press
  //--------------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    // Set the thickness
    if (d_hollowFlag) {
      d_thickness = thicknessEntry.getValue();
    } else {
      d_thickness = 0.0;
    }

    // Perform the actions
    if (e.getActionCommand() == "dist") {
      distributeParticles();
    } else if (e.getActionCommand() == "periodic")  {
      periodicParticleDist();
    } else if (e.getActionCommand() == "save")  {
      saveParticleDist();
    }
  }

  //--------------------------------------------------------------------------
  // Method for distributing particles
  //--------------------------------------------------------------------------
  private void distributeParticles() {

    // Clean the particle diameter vectors etc. and start afresh
    d_partList.clear();

    // Estimate the number of particles of each size in the RVE
    estimateRVEPartSizeDist();

    // Distribute the particles in the boxes based on the type of 
    // particles
    switch (d_partTypeFlag) {
    case CIRCLE:
      distributeCircles();
      break;
    case SPHERE:
      distributeSpheres();
      break;
    }
  }

  //--------------------------------------------------------------------------
  // Estimate the number of particles of each size in the RVE
  //--------------------------------------------------------------------------
  private void estimateRVEPartSizeDist() {

    // Update rvePartSizeDist and sideLength
    d_rvePartSizeDist.copy(d_partSizeDist);
    d_rveSize = (double) rveSizeEntry.getValue();
    d_parent.setRVESize(d_rveSize);
    d_partList.setRVESize(d_rveSize);

    int nofSizes = d_partSizeDist.nofSizesCalc;
    double[] dia = new double[nofSizes];
    double[] vol = new double[nofSizes];
    int[] num = new int[nofSizes];
    int[] scaledNum = new int[nofSizes];

    double rveSize = d_rveSize;
    double vf = d_partSizeDist.volFracInComposite*0.01;

    double totvol = 0.0;
    double volInputRVE = 0.0;
    double volActualRVE = 0.0;
    double scalefac = 1.0;

    switch (d_partTypeFlag) {
    case CIRCLE:

      // Compute area occupied by particles
      volActualRVE = rveSize*rveSize;
      for (int ii = 0; ii < nofSizes; ++ii) {
        num[ii] = d_partSizeDist.freq2DCalc[ii];
        dia[ii] = d_partSizeDist.sizeCalc[ii];
        vol[ii] = 0.25*Math.PI*dia[ii]*dia[ii];
        totvol += (num[ii]*vol[ii]);
      }
      break;

    case SPHERE:

      // Compute area occupied by particles
      volActualRVE = rveSize*rveSize*rveSize;
      for (int ii = 0; ii < nofSizes; ++ii) {
        num[ii] = d_partSizeDist.freq3DCalc[ii];
        dia[ii] = d_partSizeDist.sizeCalc[ii];
        vol[ii] = Math.PI*dia[ii]*dia[ii]*dia[ii]/6.0;
        totvol += (num[ii]*vol[ii]);
      }
      break;
    }

    // Compute volume  of input RVE and Compute scaling factor
    volInputRVE =  totvol/vf;
    scalefac = volActualRVE/volInputRVE;

    //System.out.println("Tot Vol = "+totvol+" Vf = " +vf +
    //                   " Vol Inp RVE = "+volInputRVE +
    //                   " Vol Act RVE = "+volActualRVE +
    //                   " Scale Fac = "+scalefac);

    // Compute scaled number for each size
    totvol = 0.0;
    for (int ii = 0; ii < nofSizes; ++ii) {
      scaledNum[ii] = (int) Math.round((double) num[ii]*scalefac);
      d_rvePartSizeDist.freq2DCalc[ii] = scaledNum[ii];
      d_rvePartSizeDist.freq3DCalc[ii] = scaledNum[ii];
      totvol += (scaledNum[ii]*vol[ii]);
    }

    // Compute new volume frac for each size
    for (int ii = 0; ii < nofSizes; ii++) {
      double volFrac = 100.0*vol[ii]*scaledNum[ii]/totvol;
      d_rvePartSizeDist.volFrac2DCalc[ii] = volFrac;
      d_rvePartSizeDist.volFrac3DCalc[ii] = volFrac;
    }

    // Print the update particle size distribution
    d_rvePartSizeDist.print();

    // Compute volume fraction occupied by the particles in the
    // compute distribution
    double newVolFrac = 0.0;
    switch (d_partTypeFlag) {
    case CIRCLE:
      newVolFrac = totvol/(rveSize*rveSize);
      break;
    case SPHERE:
      newVolFrac = totvol/(rveSize*rveSize*rveSize);
      break;
    }
    System.out.println("Updated volume fraction = "+newVolFrac);
  }

  //--------------------------------------------------------------------------
  // Distribute circles (distribute the circular particles in a square 
  // box with the given dimensions)
  //--------------------------------------------------------------------------
  private void distributeCircles() {

    try {
      // Create a random number generator
      Random rand = new Random();
      //final int MAX_ITER = 10000;
      final int MAX_ITER = 1000;

      // Rotation and material code are zero
      int matCode = 0;

      // Pick up each particle and place in the square ..  the largest 
      // particles first
      int nofSizesCalc = d_rvePartSizeDist.nofSizesCalc;
      for (int ii = nofSizesCalc-1; ii > -1; ii--) {
        int nofParts = d_rvePartSizeDist.freq2DCalc[ii];

        double partDia = d_rvePartSizeDist.sizeCalc[ii];
        double partDiaNext = 0.0;
        if (ii == 0)
          partDiaNext = 0.5*d_rvePartSizeDist.sizeCalc[0];
        else
          partDiaNext = d_rvePartSizeDist.sizeCalc[ii-1];
          
        for (int jj = 0; jj < nofParts; jj++) {
          
          // Iterate till the particle fits in the box
          boolean fit = false;
          
          int nofIter = 0;
          while (!fit) {
            
            // Increment the iterations and quit if the MAX_ITER is exceeded
            if (nofIter > MAX_ITER) {
              if (partDia < partDiaNext) break;
              else {
                nofIter = 0;
                partDia *= 0.9;
              }
            }
            nofIter++;
            
            // Get two random numbers for the x and y and scale to get center
            double xCent = rand.nextDouble()*d_rveSize;
            double yCent = rand.nextDouble()*d_rveSize;
            Point partCent = new Point(xCent, yCent, 0.0);

            // Find if the particle fits in the box
            boolean boxFit = isCircleInsideRVE(partDia, xCent, yCent);

            // Find if the particle intersects other particles already
            // placed in the box
            if (boxFit) {
              int nofPartsInVector = d_partList.size();
              boolean circlesIntersect = false;
              for (int kk = 0; kk < nofPartsInVector; kk++) {
                Particle part = d_partList.getParticle(kk);
                double dia1 = 2.0*part.getRadius();
                Point cent1 = part.getCenter();
                circlesIntersect = doCirclesIntersect(dia1, cent1, partDia, 
                                                      partCent);
                if (circlesIntersect) break;
              } 
              if (circlesIntersect) fit = false;
              else {

		
                // Add a particle to the particle list
                Particle newParticle = new Particle(0.5*partDia, d_rveSize,
                                                    d_thickness, partCent, 
                                                    matCode);
                d_partList.addParticle(newParticle);

                // Update the display
                d_parent.refreshDisplayPartLocFrame();

                // Set flag to true
                fit = true;
              }
            }
          }
        }
      }

      // Compute the volume fraction occupied by particles
      int vecSize = d_partList.size();
      double vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
        double rad = (d_partList.getParticle(ii)).getRadius();
        vol += Math.PI*rad*rad;
      }
      double volBox = d_rveSize*d_rveSize;
      double vfrac = vol/volBox;
      System.out.println("No of parts = "+vecSize+" Vol frac = "+vfrac);
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);

      // Fill up the rest with fines
      double partDia = d_rvePartSizeDist.sizeCalc[0];
      double fracComp = d_rvePartSizeDist.volFracInComposite/100.0;
      while (vfrac < fracComp) {

        boolean fit = false;
        int nofIter = 0;
        System.out.println("Part Dia = "+partDia+" Vol frac = "+vfrac+
          "Vol Frac Comp = "+d_rvePartSizeDist.volFracInComposite);
        while (!fit) {

          // Increment the iterations and quit if the MAX_ITER is exceeded
          if (nofIter > MAX_ITER) break;
          nofIter++;

          // Get two random numbers for the x and y and scale the co-ordinates
          double xCent = rand.nextDouble()*d_rveSize;
          double yCent = rand.nextDouble()*d_rveSize;
          Point partCent = new Point(xCent, yCent, 0.0);

          // Find if the particle fits in the box
          boolean boxFit = isCircleInsideRVE(partDia, xCent, yCent);

          // Find if the particle intersects other particles already
          // placed in the box
          if (boxFit) {
            int nofPartsInVector = d_partList.size();
            boolean circlesIntersect = false;
            for (int kk = 0; kk < nofPartsInVector; kk++) {
              Particle part = d_partList.getParticle(kk);
              double dia1 = 2.0*part.getRadius();
              Point cent1 = part.getCenter();
              circlesIntersect = doCirclesIntersect(dia1, cent1, partDia, 
                                                    partCent);
              if (circlesIntersect) break;
            } 
            if (circlesIntersect) fit = false;
            else {

              // Add a particle to the particle list
              Particle newParticle = new Particle(0.5*partDia, d_rveSize,
                                                  d_thickness, partCent, 
                                                  matCode);
              d_partList.addParticle(newParticle);
                
              // Update the display
              d_parent.refreshDisplayPartLocFrame();
                
              fit = true;
            }
          }
        }

        // Calculate the new volume
        if (fit) {
          vfrac += (0.25*Math.PI*partDia*partDia)/volBox;
        } else {
          partDia = 0.9*partDia;
        }
      }

      vecSize = d_partList.size();
      vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
        double rad = (d_partList.getParticle(ii)).getRadius();
        vol += Math.PI*rad*rad;
      }
      vfrac = vol/volBox;
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);
    } catch (Exception e) {
      System.out.println("Some exception occured in method distributeCircles");
    }
    
  }

  //--------------------------------------------------------------------------
  // Find if circles intersect
  //--------------------------------------------------------------------------
  private boolean doCirclesIntersect(double dia1, Point cent1, 
                                     double dia2, Point cent2){
    double x1 = cent1.getX();
    double y1 = cent1.getY();
    double x2 = cent2.getX();
    double y2 = cent2.getY();
    double distCent = Math.sqrt(Math.pow((x2-x1),2)+Math.pow((y2-y1),2));
    double sumRadii = dia1/2 + dia2/2;
    double gap = distCent - sumRadii;
    if (gap < 0.01*sumRadii) return true;
    //if (sumRadii > distCent) return true;
    return false;
  }

  //--------------------------------------------------------------------------
  // Find if circle is inside the RVE
  //--------------------------------------------------------------------------
  private boolean isCircleInsideRVE(double dia, double xCent, double yCent)
  {
    // Find if the particle fits in the box
    double rad = 0.5*dia;
    double xMinPartBox = xCent-rad;
    double xMaxPartBox = xCent+rad;
    double yMinPartBox = yCent-rad;
    double yMaxPartBox = yCent+rad;
    if (xMinPartBox >= 0.0 && xMaxPartBox <= d_rveSize &&
      yMinPartBox >= 0.0 && yMaxPartBox <= d_rveSize) {
      return true;
    }
    return false;
  }

  //--------------------------------------------------------------------------
  // Distribute spheres (distribute the spherical particles in a cube 
  // box with the given dimensions)
  //--------------------------------------------------------------------------
  private void distributeSpheres() {

    try {
      // Create a random number generator for the center co-ordinates
      Random rand = new Random();
      final int MAX_ITER = 30000;

      // No rotation needed; material code is 0
      double rotation = 0.0;
      int matCode = 0;

      // Pick up each particle and place in the cube ..  the largest 
      // particles first
      int nofSizesCalc = d_rvePartSizeDist.nofSizesCalc;
      for (int ii = nofSizesCalc; ii > 0; ii--) {
        int nofParts = d_rvePartSizeDist.freq3DCalc[ii-1];
        double partDia = 0.0;
        double partDiaCurr = 0.0;
        boolean fit = false;
        System.out.println("Particle size fraction # = "+ii);
        for (int jj = 0; jj < nofParts; jj++) {
          
          // Set up the particle diameter
          System.out.println("Particle # = "+jj);
          partDia = d_rvePartSizeDist.sizeCalc[ii-1];
          partDiaCurr = partDia;
          
          // Iterate till the particle fits in the box
          fit = false;
          
          int nofIter = 0;
          while (!fit) {
            
            // Increment the iterations and quit if the MAX_ITER is exceeded
            nofIter++;
            if (nofIter > MAX_ITER) return;

            // Get three random numbers for the x,y and z and scale
            double xCent = rand.nextDouble()*d_rveSize;
            double yCent = rand.nextDouble()*d_rveSize;
            double zCent = rand.nextDouble()*d_rveSize;
            Point partCent = new Point(xCent, yCent, zCent);

            // Find if the particle fits in the box
            boolean boxFit = isSphereInsideRVE(partDia, xCent, yCent, zCent);

            // Find if the particle intersects other particles already
            // placed in the box
            if (boxFit) {
              boolean spheresIntersect = false;
              int nofPartsInVector = d_partList.size();
              for (int kk = 0; kk < nofPartsInVector; kk++) {

                // Get the particle
                Particle part = (Particle) d_partList.getParticle(kk);
                double dia1 = 2.0*part.getRadius();
                Point cent1 = part.getCenter();
                spheresIntersect = doSpheresIntersect(dia1, cent1, partDia, 
                                                      partCent);
                if (spheresIntersect) break;
              } 
              if (spheresIntersect) fit = false;
              else {

                // Add a particle to the particle list
                Particle newParticle = new Particle(d_partTypeFlag, 0.5*partDia,
                                                    rotation, partCent, matCode,
                                                    d_thickness);
                d_partList.addParticle(newParticle);
                newParticle.print();

                // Update the display
                d_parent.refreshDisplayPartLocFrame();
                
                // if the fit is not perfect fit the remaining volume
                // again
                if (partDiaCurr != partDia) {
                  partDia = 
                    Math.pow(Math.pow(partDiaCurr,3)-Math.pow(partDia,3),
                            (1.0/3.0));
                  partDiaCurr = partDia;
                  nofIter = 0;
                  fit = false;
                } else {
                  fit = true;
                }
              }
            }
          }
        }
      }

      // calculate the volume of the particles
      int vecSize = d_partList.size();
      double vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
        double dia = 2.0*((Particle) d_partList.getParticle(ii)).getRadius();
        vol += dia*dia*dia*Math.PI/6.0;
      }
      double volBox = Math.pow(d_rveSize,3);
      double vfrac = vol/volBox;
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);

      // Fill up the rest with fines 
      double partDia = d_rvePartSizeDist.sizeCalc[0];
      double fracComp = d_rvePartSizeDist.volFracInComposite/100.0;
      while (vfrac < fracComp) {

        boolean fit = false;
        int nofIter = 0;
        System.out.println("Part Dia = "+partDia+" Vol frac = "+vfrac+
                           "Vol Frac = "+d_rvePartSizeDist.volFracInComposite);
        while (!fit) {

          // Increment the iterations and quit if the MAX_ITER is exceeded
          if (nofIter > MAX_ITER) break;
          nofIter++;

          // Get two random numbers for the x and y and scale the co-ordinates
          double xCent = rand.nextDouble()*d_rveSize;
          double yCent = rand.nextDouble()*d_rveSize;
          double zCent = rand.nextDouble()*d_rveSize;
          Point partCent = new Point(xCent, yCent, zCent);

          // Find if the particle fits in the box
          boolean boxFit = isSphereInsideRVE(partDia, xCent, yCent, zCent);

          // Find if the particle intersects other particles already
          // placed in the box
          if (boxFit) {
            int nofPartsInVector = d_partList.size();
            boolean spheresIntersect = false;
            for (int kk = 0; kk < nofPartsInVector; kk++) {

              // Get the particle
              Particle part = (Particle) d_partList.getParticle(kk);
              double dia1 = 2.0*part.getRadius();
              Point cent1 = part.getCenter();
              spheresIntersect = doSpheresIntersect(dia1, cent1, partDia, 
                                                    partCent);
              if (spheresIntersect) break;
            } 
            if (spheresIntersect) fit = false;
            else {

              // Add a particle to the particle list
              Particle newParticle = new Particle(d_partTypeFlag, 0.5*partDia,
                                                  rotation, partCent, matCode,
                                                  d_thickness);
              d_partList.addParticle(newParticle);
              //newParticle.print();

              // Update the display
              d_parent.refreshDisplayPartLocFrame();
                
              fit = true;
            }
          }
        }

        // Calculate the new volume
        if (fit) {
          vfrac += Math.pow(partDia,3)*Math.PI/(6.0*volBox);
        } else {
          partDia *= 0.9;
        }
      }
      vecSize = d_partList.size();
      vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
        double dia = 2.0*((Particle) d_partList.getParticle(ii)).getRadius();
        vol += dia*dia*dia*Math.PI/6.0;
      }
      vfrac = vol/volBox;
      System.out.println("Final values");
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);
    } catch (Exception e) {
      System.out.println("Some exception occured in method distributeSpheres");
    }
    
  }

  //--------------------------------------------------------------------------
  // Find if sphere is inside the RVE
  //--------------------------------------------------------------------------
  private boolean isSphereInsideRVE(double dia, double xCent, double yCent,
                                    double zCent) 
  {

    // Find if the particle fits in the box
    double rad = 0.5*dia;
    double xMinPartBox = xCent-rad;
    double xMaxPartBox = xCent+rad;
    double yMinPartBox = yCent-rad;
    double yMaxPartBox = yCent+rad;
    double zMinPartBox = zCent-rad;
    double zMaxPartBox = zCent+rad;
    if (xMinPartBox >= 0.0 && xMaxPartBox <= d_rveSize &&
      yMinPartBox >= 0.0 && yMaxPartBox <= d_rveSize &&
      zMinPartBox >= 0.0 && zMaxPartBox <= d_rveSize) {
      return true;
    }
    return false;
  }

  //--------------------------------------------------------------------------
  // Find if spheres intersect
  //--------------------------------------------------------------------------
  private boolean doSpheresIntersect(double dia1, Point cent1, 
                                     double dia2, Point cent2) {
    double x1 = cent1.getX(); 
    double y1 = cent1.getY(); 
    double z1 = cent1.getZ(); 
    double x2 = cent2.getX(); 
    double y2 = cent2.getY(); 
    double z2 = cent2.getZ(); 

    double distCent = 
      Math.sqrt(Math.pow((x2-x1),2)+Math.pow((y2-y1),2)+Math.pow((z2-z1),2));
    double sumRadii = dia1/2 + dia2/2;
    if (sumRadii > distCent) return true;
    return false;
  }


  //--------------------------------------------------------------------------
  // Create a periodic distribution of particles in the box.  Similar
  // approach to random sequential packing of distributeCircles
  //--------------------------------------------------------------------------
  private void periodicParticleDist() {

    final int MAX_ITER = 200000;

    // Set material code to zero
    int matCode = 0;

    // Clean the particle diameter vectors etc. and start afresh
    d_partList.clear();

    // Estimate the number of particles of each size in the RVE
    estimateRVEPartSizeDist();

    // Create a random number generator
    Random rand = new Random();

    // Get the number of particle sizes
    int nofSizesCalc = d_rvePartSizeDist.nofSizesCalc;

    // The sizes are distributed with the smallest first.  Pick up
    // the largest size and iterate down through smaller sizes
    for (int ii = nofSizesCalc; ii > 0; ii--) {

      // Get the number of particles for the current size
      int nofParts = d_rvePartSizeDist.freq2DCalc[ii-1];

      // Get the particle size
      double partRad = 0.5*d_rvePartSizeDist.sizeCalc[ii-1];

      // Increase the size of the box so that periodic distributions
      // are allowed
      double boxMin = -0.9*partRad;
      double boxMax = d_rveSize+0.9*partRad;

      // Calculate the limits of the box oustide which periodic bcs
      // come into play
      double boxInMin = partRad;
      double boxInMax = d_rveSize-partRad;
      
      // Pick up each particle and insert it into the box
      //System.out.println("No. of particles to be inserted = "+nofParts);
      for (int jj = 0; jj < nofParts; jj++) {

        boolean fit = false;
        int nofIter = 0;
        while (!fit && nofIter < MAX_ITER) {

          // Generate a random location for the particle
          // (from boxmin-0.5*partDia to boxmax+0.5*partdia)
          double tx = rand.nextDouble();
          double ty = rand.nextDouble();
          double xCent = (1-tx)*boxMin + tx*boxMax;
          double yCent = (1-ty)*boxMin + ty*boxMax;
          Point partCent = new Point(xCent, yCent, 0.0);

          // If the particle is partially outside the original box
          // then deal with it separately, otherwise do the standard checks
          if (inLimits(xCent, boxInMin, boxInMax) &&
              inLimits(yCent, boxInMin, boxInMax) ) {

            // Particle is inside the box .. find if it intersects another
            // particle previously placed in box.  If it does then 
            // try again otherwise add the particle to the list.
            if (!intersectsAnother(partRad, partCent)) {

              // Add a particle to the particle list
              Particle newParticle = new Particle(partRad, d_rveSize, 
                                                  d_thickness, partCent, 
                                                  matCode);
              d_partList.addParticle(newParticle);
              //newParticle.print();

              // Update the display
              d_parent.refreshDisplayPartLocFrame();
                
              fit = true;
            }
            ++nofIter;
          } else {

            // Check if this particle intersects another
            if (!intersectsAnother(partRad, partCent)) {

              // Particle is partially outside the box  ... create periodic 
              // images and check each one (there are eight possible locations 
              // of the // center
              double[] xLoc = new double[3];
              double[] yLoc = new double[3];
              int nofLoc = findPartLoc(partRad, xCent, yCent, 0, d_rveSize,
                                       xLoc, yLoc);

              Point cent1 = new Point(xLoc[0], yLoc[0], 0.0);
              Point cent2 = new Point(xLoc[1], yLoc[1], 0.0);
              Point cent3 = new Point(xLoc[2], yLoc[2], 0.0);

              // Carry out checks for each of the locations
              if (nofLoc != 0) {
                if (nofLoc == 3) {
                  if (!intersectsAnother(partRad, cent1)) {
                    if (!intersectsAnother(partRad, cent2)) {
                      if (!intersectsAnother(partRad, cent3)) {
                        fit = true;

                        // Add original particles to the particle list
                        Particle pOrig = new Particle(partRad, d_rveSize, 
                                                      d_thickness, partCent, 
                                                      matCode);
                        d_partList.addParticle(pOrig);
                        //pOrig.print();

                        // Add symmetry particles to the particle list
                        Particle p1 = new Particle(partRad, d_rveSize, 
                                                   d_thickness, cent1, 
                                                   matCode);
                        d_partList.addParticle(p1);
                        //p1.print();

                        Particle p2 = new Particle(partRad, d_rveSize, 
                                                   d_thickness, cent2, 
                                                   matCode);
                        d_partList.addParticle(p2);
                        //p2.print();

                        Particle p3 = new Particle(partRad, d_rveSize, 
                                                   d_thickness, cent3,
                                                   matCode);
                        d_partList.addParticle(p3);
                        //p3.print();

                        // Update the display
                        d_parent.refreshDisplayPartLocFrame();
                      }
                    }
                  }
                } else {
                  if (!intersectsAnother(partRad, cent1)) {
                    fit = true;

                    // Add original particles to the particle list
                    Particle pOrig = new Particle(partRad, d_rveSize, 
                                                  d_thickness, partCent, 
                                                  matCode);
                    d_partList.addParticle(pOrig);
                    //pOrig.print();

                    // Add symmetry particles to the particle list
                    Particle p1 = new Particle(partRad, d_rveSize, 
                                               d_thickness, cent1, 
                                               matCode);
                    d_partList.addParticle(p1);
                    //p1.print();

                    // Update the display
                    d_parent.refreshDisplayPartLocFrame();
                  }
                }
              }
            }
            ++nofIter;
          }
          if (nofIter%MAX_ITER == 0) {
            partRad *= 0.99;
            //System.out.println("No. of Iterations = " + nofIter +
            //                   " Particle Radius = " + partRad);
          }
        }
        //System.out.println("Particle No = " + jj);
      }
      //System.out.println("Particle Size No = " + ii);
    }
  }

  private boolean inLimits(double x, double min, double max) {
    if (x == min || x == max || (x > min && x < max)) return true;
    return false;
  }

  private boolean intersectsAnother(double radius, Point center) {
    int nofParts = d_partList.size();
    for (int kk = 0; kk < nofParts; kk++) {
      Particle part = (Particle) d_partList.getParticle(kk);
      double dia1 = 2.0*part.getRadius();
      Point cent1 = part.getCenter();
      if (doCirclesIntersect(dia1, cent1, radius*2.0, center))
        return true;
    }
    return false;
  }
 
  // Return the number of new locations
  private int findPartLoc(double rad, double x, double y, double min, 
                          double max, double[] xLoc, double[] yLoc) {
    // Create a box around the particle
    double xmin = x - rad;
    double xmax = x + rad;
    double ymin = y - rad;
    double ymax = y + rad;

    // Check the 8 regions to see if the particles intersect any of these
    // regions .. first the corners and then the sides
    if (xmin < min && ymin < min) {
      // Create three more particles at the other three corners
      // This is the lower left hand corner
      // New Particle 1 : lower right hand
      xLoc[0] = x + d_rveSize;
      yLoc[0] = y;
      // New Particle 2 : upper right hand
      xLoc[1] = x + d_rveSize;
      yLoc[1] = y + d_rveSize;
      // New Particle 3 : upper left hand
      xLoc[2] = x;
      yLoc[2] = y + d_rveSize;
      return 3;
    }
    if (xmax > max && ymin < min) {
      // Create three more particles at the other three corners
      // This is the lower right hand corner
      // New Particle 1 : lower left hand
      xLoc[0] = x - d_rveSize;
      yLoc[0] = y;
      // New Particle 2 : upper right hand
      xLoc[1] = x;
      yLoc[1] = y + d_rveSize;
      // New Particle 3 : upper left hand
      xLoc[2] = x - d_rveSize;
      yLoc[2] = y + d_rveSize;
      return 3;
    }
    if (xmax > max && ymax > max) {
      // Create three more particles at the other three corners
      // This is the upper right hand corner
      // New Particle 1 : lower left hand
      xLoc[0] = x - d_rveSize;
      yLoc[0] = y - d_rveSize;
      // New Particle 2 : lower right hand
      xLoc[1] = x;
      yLoc[1] = y - d_rveSize;
      // New Particle 3 : upper left hand
      xLoc[2] = x - d_rveSize;
      yLoc[2] = y;
      return 3;
    }
    if (xmin < min && ymax > max) {
      // Create three more particles at the other three corners
      // This is the upper left hand corner
      // New Particle 1 : lower left hand
      xLoc[0] = x;
      yLoc[0] = y - d_rveSize;
      // New Particle 2 : lower right hand
      xLoc[1] = x + d_rveSize;
      yLoc[1] = y - d_rveSize;
      // New Particle 3 : upper right hand
      xLoc[2] = x + d_rveSize;
      yLoc[2] = y;
      return 3;
    }
    if (xmin < min) {
      // Create one more particles at right side
      // This is the left side
      // New Particle 1 : right side
      xLoc[0] = x + d_rveSize;
      yLoc[0] = y;
      return 1;
    }
    if (xmax > max) {
      // Create one more particles at left side
      // This is the right side
      // New Particle 1 : left side
      xLoc[0] = x - d_rveSize;
      yLoc[0] = y;
      return 1;
    }
    if (ymin < min) {
      // Create one more particles at upper side
      // This is the lower side
      // New Particle 1 : upper side
      xLoc[0] = x;
      yLoc[0] = y + d_rveSize;
      return 1;
    }
    if (ymax > max) {
      // Create one more particles at bottom side
      // This is the top side
      // New Particle 1 : bottom side
      xLoc[0] = x;
      yLoc[0] = y - d_rveSize;
      return 1;
    }
    return 0;
  }

  // Method for saving computed particle distribution
  private void saveParticleDist() {

    // Get the name of the file
    File file = null;
    JFileChooser fc = new JFileChooser(new File(".."));
    int returnVal = fc.showSaveDialog(this);
    if (returnVal == JFileChooser.APPROVE_OPTION) file = fc.getSelectedFile();
    if (file == null) return;

    // Save to file
    d_partList.saveToFile(file, d_partTypeFlag);
  }

}
