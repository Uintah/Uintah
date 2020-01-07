//**************************************************************************
// Program : GenerateParticlePanel.java
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
import java.util.Vector;
import java.io.*;
import javax.swing.*;

//**************************************************************************
// Class   : GenerateParticlePanel
// Purpose : Generate the locations of randomly distributed particles of a
//           particular size distribution.
//**************************************************************************
public class GenerateParticlePanel extends JPanel {

  // Data that are stored for the life of the object
  private DecimalField nofPartEntry = null;
  private IntegerField nofCells = null;
  private JComboBox partTypeCBox = null;
  private JComboBox roundOffCBox = null;
  private JButton distribButton = null;
  private JButton moveButton = null;
  private JButton gridButton = null;
  private TopCanvas topCanvas = null;
  //private SideCanvas sideCanvas = null;
  //private FrontCanvas frontCanvas = null;
  //private OrthoCanvas orthoCanvas = null;
  private JButton saveButton = null;
  private JButton closeButton = null;

  // Other data
  private Vector d_boxList = null; // A Vector of boxes
  private ParticleSize d_partSizeDist = null;
  private double d_sideLength = 0.0;
  private int partTypeFlag = 0;
  private int roundOffFlag = 0;
  private double len2DSquare = 0.0;
  private double len2DCircle = 0.0;
  private double len3DCube = 0.0;
  private double len3DSphere = 0.0;
  private Vector partDiaVector = null;
  private Vector partXCentVector = null;
  private Vector partYCentVector = null;
  private Vector partZCentVector = null;
  private Vector partSideVector = null;
  private Vector partRotVector = null;
  private Vector partPt1Vector = null;
  private Vector partPt2Vector = null;
  private Vector partPt3Vector = null;
  private Vector partPt4Vector = null;

  // static data
  public static final int TOP = 1;
  public static final int SIDE = 2;
  public static final int FRONT = 3;
  public static final int CIRCLE = 1;
  public static final int SQUARE = 2;
  public static final int SPHERE = 3;
  public static final int CUBE = 4;
  public static final int YES = 1;
  public static final int NO = 2;
  
  public GenerateParticlePanel() {

    // set the size
    //setSize(300,300);
    setLocation(500,50);
    setBackground(new Color(170,170,170));

    //setTitle("Generate Particle Location Distribution");
    partTypeFlag = CIRCLE;
    roundOffFlag = YES;
    d_sideLength = 100;

    // Create a vector to store the already fitted particles
    partDiaVector = new Vector();
    partXCentVector = new Vector();
    partYCentVector = new Vector();
    partZCentVector = new Vector();
    partSideVector = new Vector();
    partRotVector = new Vector();
    partPt1Vector = new Vector();
    partPt2Vector = new Vector();
    partPt3Vector = new Vector();
    partPt4Vector = new Vector();

    // Create a panels to contain the components
    JPanel panel1 = new JPanel(new GridLayout(4,0));
    JPanel panel10 = new JPanel(new GridLayout(4,0));
    JPanel panel2 = new JPanel();
    //JPanel panel2 = new JPanel(new GridLayout(2,2));
    JPanel panel3 = new JPanel(new GridLayout(1,0));

    // Create the components
    JLabel label0 = new JLabel("No. of Grid Cells (in one direction) ");
    nofCells = new IntegerField(1000,5);
    JLabel label1 = new JLabel("No. of particles (x 100)");
    nofPartEntry = new DecimalField(1.0,3);
    JLabel label2 = new JLabel("Type of particle ");
    partTypeCBox = new JComboBox();
    partTypeCBox.addItem("2D-Circle");
    partTypeCBox.addItem("3D-Sphere");
    partTypeCBox.addItem("2D-Square");
    partTypeCBox.addItem("3D-Cube");
    JLabel label3 = new JLabel("Round off particle size to nearest 10");
    roundOffCBox = new JComboBox();
    roundOffCBox.addItem("Yes");
    roundOffCBox.addItem("No");
    JLabel labelrun = new JLabel("Click on one of the following three buttons");
    distribButton = new JButton("Create Random Distribution");
    distribButton.setActionCommand("dist");
    moveButton = new JButton("Create Periodic Distribution");
    moveButton.setActionCommand("movedist");
    //moveButton = new JButton("Distribute Particles (MC Motion)");
    //moveButton.setActionCommand("movedist");
    gridButton = new JButton("Create Filled GMC Grid");
    gridButton.setActionCommand("grid");
    topCanvas = new TopCanvas(400,400);
    //sideCanvas = new SideCanvas(300,300);
    //frontCanvas = new FrontCanvas(300,300);
    //orthoCanvas = new OrthoCanvas(300,300);
    saveButton = new JButton("Save Calculated Data");
    saveButton.setActionCommand("save");
    closeButton = new JButton("Close this Window");
    closeButton.setActionCommand("close");

    // Add the components to the panels
    panel1.add(label0);
    panel1.add(nofCells);
    panel1.add(label1);
    panel1.add(nofPartEntry);
    panel1.add(label2);
    panel1.add(partTypeCBox);
    panel1.add(label3);
    panel1.add(roundOffCBox);
    panel10.add(labelrun);
    panel10.add(distribButton);
    panel10.add(moveButton);
    panel10.add(gridButton);

    panel2.add(topCanvas);
    //panel2.add(orthoCanvas);
    //panel2.add(frontCanvas);
    //panel2.add(sideCanvas);

    panel3.add(saveButton);
    panel3.add(closeButton);

    // Create a gridbag and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
				    1.0,1.0, 0,0, 1,1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
				    1.0,1.0, 0,1, 1,1, 5);
    gb.setConstraints(panel10, gbc);
    add(panel10);
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
				    1.0,1.0, 0,2, 1,1, 5);
    gb.setConstraints(panel2, gbc);
    add(panel2);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
				    1.0,1.0, 0,3, 1,1, 5);
    gb.setConstraints(panel3, gbc);
    add(panel3);

    // Create and add the listeners
    ComboBoxListener comboBoxListener = new ComboBoxListener();
    partTypeCBox.addItemListener(comboBoxListener);
    roundOffCBox.addItemListener(comboBoxListener);
    ButtonListener buttonListener = new ButtonListener();
    distribButton.addActionListener(buttonListener);
    moveButton.addActionListener(buttonListener);
    gridButton.addActionListener(buttonListener);
    saveButton.addActionListener(buttonListener);
    closeButton.addActionListener(buttonListener);

    // Initialize the subcell list and box list
    d_boxList = new Vector();
  }

  // Method for setting the particle distribution
  public void setParticleSizeDist(ParticleSize ps) {
    d_partSizeDist = ps;
  }

  //**************************************************************************
  // Class   : ComboBoxListener
  // Purpose : Listens for item picked in combo box and takes action as
  //           required.
  //**************************************************************************
  private class ComboBoxListener implements ItemListener {
    public void itemStateChanged(ItemEvent e) {

      // Get the item that has been selected
      String item = String.valueOf(e.getItem());
      if (item == "2D-Circle") {
	partTypeFlag = CIRCLE;
      } else if (item == "2D-Square") {
	partTypeFlag = SQUARE;
      } else if (item == "3D-Sphere") {
	partTypeFlag = SPHERE;
      } else if (item == "3D-Cube") {
	partTypeFlag = CUBE;
      } 
      if (item == "Yes") {
	roundOffFlag = YES;
      } else if (item == "No") {
	roundOffFlag = NO;
      }
    }
  }

  //**************************************************************************
  // Class   : ButtonListener
  // Purpose : Listens for action on the button picked and takes action
  //           required.
  //**************************************************************************
  private class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {

      if (e.getActionCommand() == "dist") 
	distributeParticles();
      else if (e.getActionCommand() == "movedist") 
	//distributeMoveParticles();
	periodicParticleDist();
      else if (e.getActionCommand() == "grid") 
	createAndFillGMCGrid();
      else if (e.getActionCommand() == "save") 
	saveParticleDist();
      else 
	setVisible(false);
    }
  }

  //--------------------------------------------------------------------------
  // Create a GMC Grid and fill it up with particles based on input
  // distribution
  // Currently only for 2D Squares (in future also for 3D Cubes)
  // This mode ignores the number of particles that is input and calculates
  // the number required based on the size of the grid with the smallest
  // filling exactly one cell completely.
  //--------------------------------------------------------------------------
  private void createAndFillGMCGrid() {

    if (partTypeFlag != SQUARE) return;

    // Get the number of grid cells
    int nofDivs = nofCells.getValue();

    // Calculate the sizes of the particle templates
    double minSize = calculateTemplateSizes();

    // Calculate the size of the gridded domain 
    double domainSize = minSize*(double)nofDivs;
    d_sideLength = domainSize;
    System.out.println("DomainSize = "+domainSize+" minSize = "+minSize);

    // Calculate the frequency of occurrence of each particle size
    calculateFrequency(minSize, domainSize);

    // Create the Grid (SubcellList) explicitly and place templates 
    // in each subcell
    //placeInGMCGrid(nofDivs, minSize);

    // Repaint the Front (transverse) canvas
    topCanvas.repaint();
    //sideCanvas.repaint();
    //frontCanvas.repaint();
  }

  //--------------------------------------------------------------------------
  // Calculate the template sizes (based on the particle sizes)
  //--------------------------------------------------------------------------
  private double calculateTemplateSizes() {

    // If the particle size is to rounded to the nearest 10, do the 
    // rounding
    int nofSizesCalc = d_partSizeDist.nofSizesCalc;
    //if (roundOffFlag == YES) {
     // for (int ii = 0; ii < nofSizesCalc; ii++) {
	//double size = d_partSizeDist.sizeCalc[ii];
	//int roundedSize = 10*Math.round((float)(size/10.0));
	//d_partSizeDist.sizeCalc[ii] = (double) roundedSize;
	// System.out.println("Size = "+d_partSizeDist.sizeCalc[ii]);
      //}
    //}

    // Get the smallest sized particle
    double minSize = 1.0e10;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      if (d_partSizeDist.sizeCalc[ii] < minSize)
	minSize = d_partSizeDist.sizeCalc[ii];
    }

    return minSize;
  }

  //--------------------------------------------------------------------------
  // Calculate the new frequency of the particle distribution
  //--------------------------------------------------------------------------
  private void calculateFrequency(double minSize, double domainSize) {

    // Find the total volume (area in 2D) occupied by the templates
    // based on 100 particles
    int nofSizesCalc = d_partSizeDist.nofSizesCalc;
    double vol = 0.0;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      double radius = d_partSizeDist.sizeCalc[ii]*0.5;
      int frequency = d_partSizeDist.freq2DCalc[ii];
      vol += Math.PI*Math.pow(radius,2)* (double) frequency;
    }
    System.out.println("Vol of particle = "+vol);

    // This volume/area has to be the required volume/area fraction
    double volFrac = d_partSizeDist.volFracInComposite/100.0;
    double boxVol = vol/volFrac;
    System.out.println("Vol of Box containing particles = "+boxVol);

    // Find the box scaling factor
    double scaleFactor = Math.pow(domainSize,2)/boxVol;
    System.out.println("Scale factor = "+scaleFactor);

    // Scale the frequency of the particles based on this scale factor
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      double number = (double) d_partSizeDist.freq2DCalc[ii];
      number *= scaleFactor;
      d_partSizeDist.freq2DCalc[ii] = (int) number;
    }

    // Check the volume fraction
    vol = 0.0;
    int zero = 0;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      double radius = d_partSizeDist.sizeCalc[ii]*0.5;
      int frequency = d_partSizeDist.freq2DCalc[ii];
      if (frequency == 0 && zero == 0) zero = ii;
      vol += Math.PI*Math.pow(radius,2)* (double) frequency;
      System.out.println("Size = "+radius+" # = "+frequency);
    }
    double totVol = Math.pow(domainSize, 2);
    double newVolFrac = vol/totVol;
    System.out.println("Volume fraction = "+newVolFrac);

    // Add one particle of higher size and check VF
    double newVol = vol;
    int ballSizeID = zero;
    do {
      double radius = 0.5* (double) d_partSizeDist.sizeCalc[ballSizeID];
      newVol += Math.PI*Math.pow(radius,2);
      d_partSizeDist.freq2DCalc[ballSizeID] = 1;
      newVolFrac = newVol/totVol;
      System.out.println("Size = "+radius+" # = "+d_partSizeDist.freq2DCalc[ballSizeID]);
      ++ballSizeID;
    } while (newVolFrac < volFrac && ballSizeID < nofSizesCalc ) ;
    System.out.println("Volume fraction = "+newVolFrac);
  }

  //--------------------------------------------------------------------------
  // Method for distributing particles
  //--------------------------------------------------------------------------
  private void distributeParticles() {

    // Calculate the size of the box containing the particles
    calcSizeofBox();

    // Distribute the particles in the boxes based on the type of 
    // particles
    switch (partTypeFlag) {
    case CIRCLE:
      distributeCircles();
      break;
    case SQUARE:
      distributeSquares();
      break;
    case SPHERE:
      distributeSpheres();
      break;
    case CUBE:
      distributeCubes();
      break;
    }
    
  }

  // Method for distributing and moving particles using MC
  private void distributeMoveParticles() {

    // Calculate the size of the box containing the particles
    calcSizeofBox();

    // Distribute the particles in the boxes based on the type of 
    // particles
    Packing pack = null;
    int nofParticles = 0;
    switch (partTypeFlag) {
    case CIRCLE:
      pack = new Packing(Packing.TWO_DIM, d_partSizeDist);
      pack.createPacking(10000, 10000, 92.0);
      nofParticles = pack.nofParticles();
      for (int ii = 0; ii < nofParticles; ii++) {
	partDiaVector.addElement(new Double(2.0*pack.radius(ii)));
	partXCentVector.addElement(new Double(pack.xLoc(ii)));
	partYCentVector.addElement(new Double(pack.yLoc(ii)));
      }
      d_sideLength = pack.boxSize();
      break;
    case SPHERE:
      pack = new Packing(Packing.THREE_DIM, d_partSizeDist);
      pack.createPacking(10000, 10000, 92.0);
      nofParticles = pack.nofParticles();
      for (int ii = 0; ii < nofParticles; ii++) {
	partDiaVector.addElement(new Double(2.0*pack.radius(ii)));
	partXCentVector.addElement(new Double(pack.xLoc(ii)));
	partYCentVector.addElement(new Double(pack.yLoc(ii)));
	partZCentVector.addElement(new Double(pack.zLoc(ii)));
      }
      d_sideLength = pack.boxSize();
      break;
    default:
      break;
    }
    // Repaint the top canvas
    topCanvas.repaint();
    //sideCanvas.repaint();
    //frontCanvas.repaint();
  }

  // Calculate the size of the box
  private void calcSizeofBox() {

    // Get the total number of particles to be used from the input
    double nofPartScaleFactor = (double) nofPartEntry.getValue();
    if (nofPartScaleFactor <= 0.0) nofPartScaleFactor = 1.0;
    int nofSizesCalc = d_partSizeDist.nofSizesCalc;

    // Scale all the numbers so that they are with respect to 100 particles
    // The number of particles is either 100 or 100,000
    // (***** Just do 2D for now *****)
    double[] partDia = new double[nofSizesCalc]; 
    double[] nofPart = new double[nofSizesCalc];
    double totPart = 0;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      partDia[ii] = d_partSizeDist.sizeCalc[ii];
      nofPart[ii] = (double) d_partSizeDist.freq2DCalc[ii];
      totPart += nofPart[ii];
    }
    // If the scale factor is not 1 then scale up the nummber of particles
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      nofPart[ii] *= nofPartScaleFactor*100.0/totPart;
      System.out.println(partDia[ii]+"    "+nofPart[ii]);
    }
    // Loop thru the particles and form a new list
    double fracPart = 0.0;
    double fracDia = 0.0;
    int jj = nofSizesCalc-1;
    int count = 0;
    int nofNewParts = 0;
    double[] dia = new double[10];
    double[] num = new double[10];
    while (nofPart[jj] < 1.0) {
      if (fracPart < 1.0) {
        fracPart += nofPart[jj];
	fracDia += partDia[jj];
	++count; 
      } else {
	if (count > 0) {
          fracDia /= count;
	  dia[nofNewParts] = fracDia;
	  num[nofNewParts] = fracPart;
	  ++nofNewParts;
	  fracDia = 0.0;
	  fracPart = 0.0;
	  count = 0;
	}
      }
      --jj;
    }
    if (count > 0) {
      fracDia /= count;
      dia[nofNewParts] = fracDia;
      num[nofNewParts] = fracPart;
      ++nofNewParts;
    }
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      if (nofPart[ii] < 1.0) {
	nofSizesCalc = ii + nofNewParts;
	for (int kk = 0; kk < nofNewParts; ++kk) {
	  partDia[ii+kk] = dia[nofNewParts-kk-1]; 
	  nofPart[ii+kk] = num[nofNewParts-kk-1]; 
	}
	break;
      }
    }
    d_partSizeDist.nofSizesCalc = nofSizesCalc;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      d_partSizeDist.freq2DCalc[ii] = (int) Math.round(nofPart[ii]);
      d_partSizeDist.sizeCalc[ii] = partDia[ii];
      d_partSizeDist.freq3DCalc[ii] = (int) Math.round(nofPart[ii]);
      System.out.println(d_partSizeDist.sizeCalc[ii] + "   " +
			 d_partSizeDist.freq2DCalc[ii]);
    }

    // If the particle size is to rounded to the nearest 10, do the 
    // rounding
    if (roundOffFlag == YES) {
      for (int ii = 0; ii < nofSizesCalc; ii++) {
	double size = d_partSizeDist.sizeCalc[ii];
	int roundedSize = 10*Math.round((float)(size/10.0));
	d_partSizeDist.sizeCalc[ii] = (double) roundedSize;
	// System.out.println("Size = "+d_partSizeDist.sizeCalc[ii]);
      }
    }

    // Calculate the volume occupied by the particles in 2D and 3D
    // assuming they are square or cubic (after scaling with the 
    // nofPartScaleFactor)
    double vol2DSquare = 0.0;
    double vol3DCube = 0.0;
    for (int ii = 0; ii < nofSizesCalc; ii++) {
      double size = d_partSizeDist.sizeCalc[ii];
      double number2D = (double) d_partSizeDist.freq2DCalc[ii];
      double number3D = (double) d_partSizeDist.freq3DCalc[ii];
      vol2DSquare += number2D*Math.pow(size,2);
      vol3DCube += number3D*Math.pow(size,3);
    }

    // Now calculate the volume if the particles are circles or spheres
    double vol2DCircle = (Math.PI/4)*vol2DSquare;
    double vol3DSphere = (Math.PI/6)*vol3DCube;
    // System.out.println("2D Vol Sq = "+vol2DSquare+" 3D Vol = "+ vol3DCube);
    // System.out.println("2D Vol Ci = "+vol2DCircle+" 3D Vol = "+ vol3DSphere);

    // Calculate the volume of the square or cube containing each of these
    // volumes of particles based on the particulate volume in composite
    double volFrac = d_partSizeDist.volFracInComposite;
    vol2DSquare *= (100.0/volFrac);
    vol3DCube *= (100.0/volFrac);
    vol2DCircle *= (100.0/volFrac);
    vol3DSphere *= (100.0/volFrac);
    // System.out.println("2D Vol Sq = "+vol2DSquare+" 3D Vol = "+ vol3DCube);
    // System.out.println("2D Vol Ci = "+vol2DCircle+" 3D Vol = "+ vol3DSphere);

    // Calculate the length of the side of the square or cube
    len2DSquare = Math.pow(vol2DSquare,0.5);
    len2DCircle = Math.pow(vol2DCircle,0.5);
    len3DCube = Math.pow(vol3DCube,(1.0/3.0));
    len3DSphere = Math.pow(vol3DSphere,(1.0/3.0));
    System.out.println("2D Len Sq = "+len2DSquare+" 3D Len = "+ len3DCube);
    System.out.println("2D Len Ci = "+len2DCircle+" 3D Len = "+ len3DSphere);
  }

  // Create a periodic distribution of particles in the box.  Similar
  // approach to random sequential packing of distributeCircles
  private void periodicParticleDist() {

    final int MAX_ITER = 200000;

    // Clean the particle diameter vectors etc. and start afresh
    if (!partDiaVector.isEmpty()) {
      partDiaVector.clear(); 
      partXCentVector.clear(); 
      partYCentVector.clear(); 
    }

    // Calculate the size of the box containing the particles
    calcSizeofBox();

    // Create a random number generator
    Random rand = new Random();

    // The length of a side of the box
    d_sideLength = len2DCircle;

    // Get the number of particle sizes
    int nofSizesCalc = d_partSizeDist.nofSizesCalc;

    // The sizes are distributed with the smallest first.  Pick up
    // the largest size and iterate down through smaller sizes
    for (int ii = nofSizesCalc; ii > 0; ii--) {

      // Get the number of particles for the current size
      int nofParts = d_partSizeDist.freq2DCalc[ii-1];

      // Get the particle size
      double partRad = d_partSizeDist.sizeCalc[ii-1]/2;

      // Increase the size of the box so that periodic distributions
      // are allowed
      double boxMin = -0.9*partRad;
      double boxMax = d_sideLength+0.9*partRad;

      // Calculate the limits of the box oustide which periodic bcs
      // come into play
      double boxInMin = partRad;
      double boxInMax = d_sideLength-partRad;
      
      // Pick up each particle and insert it into the box
      System.out.println("No. of particles to be inserted = "+nofParts);
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

	  // If the particle is partially outside the original box
	  // then deal with it separately, otherwise do the standard checks
	  if (inLimits(xCent, boxInMin, boxInMax) &&
	      inLimits(yCent, boxInMin, boxInMax) ) {

	    // Particle is inside the box .. find if it intersects another
	    // particle previously placed in box.  If it does then 
	    // try again otherwise add the particle to the list.
	    if (!intersectsAnother(partRad, xCent, yCent)) {
	      fit = true;
	      partDiaVector.addElement(new Double(partRad*2.0));
	      partXCentVector.addElement(new Double(xCent));
	      partYCentVector.addElement(new Double(yCent));
	    }
	    ++nofIter;
	  } else {

	    // Check if this particle intersects another
	    if (!intersectsAnother(partRad, xCent, yCent)) {

	      // Particle is partially outside the box  ... create periodic images
	      // and check each one (there are eight possible locations of the
	      // center
	      double[] xLoc = new double[3];
              double[] yLoc = new double[3];
	      //int nofLoc = findPartLoc(partRad, xCent, yCent, boxInMin, boxInMax,
		//		       xLoc, yLoc);
	      int nofLoc = findPartLoc(partRad, xCent, yCent, 0, d_sideLength,
				       xLoc, yLoc);

	      // Carry out checks for each of the locations
	      if (nofLoc != 0) {
		if (nofLoc == 3) {
		  if (!intersectsAnother(partRad, xLoc[0], yLoc[0])) {
		    if (!intersectsAnother(partRad, xLoc[1], yLoc[1])) {
		      if (!intersectsAnother(partRad, xLoc[2], yLoc[2])) {
			fit = true;
			partDiaVector.addElement(new Double(partRad*2.0));
			partXCentVector.addElement(new Double(xCent));
			partYCentVector.addElement(new Double(yCent));
			partDiaVector.addElement(new Double(partRad*2.0));
			partXCentVector.addElement(new Double(xLoc[0]));
			partYCentVector.addElement(new Double(yLoc[0]));
			partDiaVector.addElement(new Double(partRad*2.0));
			partXCentVector.addElement(new Double(xLoc[1]));
			partYCentVector.addElement(new Double(yLoc[1]));
			partDiaVector.addElement(new Double(partRad*2.0));
			partXCentVector.addElement(new Double(xLoc[2]));
			partYCentVector.addElement(new Double(yLoc[2]));
		      }
		    }
		  }
		} else {
		  if (!intersectsAnother(partRad, xLoc[0], yLoc[0])) {
		    fit = true;
		    partDiaVector.addElement(new Double(partRad*2.0));
		    partXCentVector.addElement(new Double(xCent));
		    partYCentVector.addElement(new Double(yCent));
		    partDiaVector.addElement(new Double(partRad*2.0));
		    partXCentVector.addElement(new Double(xLoc[0]));
		    partYCentVector.addElement(new Double(yLoc[0]));
		  }
		}
	      }
	    }
	    ++nofIter;
	  }
	  if (nofIter%MAX_ITER == 0) {
	    partRad *= 0.99;
	    System.out.println("No. of Iterations = " + nofIter +
			       " Particle Radius = " + partRad);
	  }
	}
	System.out.println("Particle No = " + jj);
      }
      System.out.println("Particle Size No = " + ii);
    }
    topCanvas.repaint();
  }

  private boolean inLimits(double x, double min, double max) {
    if (x == min || x == max || (x > min && x < max)) return true;
    return false;
  }

  private boolean intersectsAnother(double rad, double xCent, double yCent) {
    int nofParts = partDiaVector.size();
    double dia1, x1, y1;
    for (int kk = 0; kk < nofParts; kk++) {
      dia1 = ((Double) (partDiaVector.elementAt(kk))).doubleValue();
      x1 = ((Double) (partXCentVector.elementAt(kk))).doubleValue();
      y1 = ((Double) (partYCentVector.elementAt(kk))).doubleValue();
      if (doCirclesIntersect(dia1, x1, y1, rad*2.0, xCent, yCent))
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
      xLoc[0] = x + d_sideLength;
      yLoc[0] = y;
      // New Particle 2 : upper right hand
      xLoc[1] = x + d_sideLength;
      yLoc[1] = y + d_sideLength;
      // New Particle 3 : upper left hand
      xLoc[2] = x;
      yLoc[2] = y + d_sideLength;
      return 3;
    }
    if (xmax > max && ymin < min) {
      // Create three more particles at the other three corners
      // This is the lower right hand corner
      // New Particle 1 : lower left hand
      xLoc[0] = x - d_sideLength;
      yLoc[0] = y;
      // New Particle 2 : upper right hand
      xLoc[1] = x;
      yLoc[1] = y + d_sideLength;
      // New Particle 3 : upper left hand
      xLoc[2] = x - d_sideLength;
      yLoc[2] = y + d_sideLength;
      return 3;
    }
    if (xmax > max && ymax > max) {
      // Create three more particles at the other three corners
      // This is the upper right hand corner
      // New Particle 1 : lower left hand
      xLoc[0] = x - d_sideLength;
      yLoc[0] = y - d_sideLength;
      // New Particle 2 : lower right hand
      xLoc[1] = x;
      yLoc[1] = y - d_sideLength;
      // New Particle 3 : upper left hand
      xLoc[2] = x - d_sideLength;
      yLoc[2] = y;
      return 3;
    }
    if (xmin < min && ymax > max) {
      // Create three more particles at the other three corners
      // This is the upper left hand corner
      // New Particle 1 : lower left hand
      xLoc[0] = x;
      yLoc[0] = y - d_sideLength;
      // New Particle 2 : lower right hand
      xLoc[1] = x + d_sideLength;
      yLoc[1] = y - d_sideLength;
      // New Particle 3 : upper right hand
      xLoc[2] = x + d_sideLength;
      yLoc[2] = y;
      return 3;
    }
    if (xmin < min) {
      // Create one more particles at right side
      // This is the left side
      // New Particle 1 : right side
      xLoc[0] = x + d_sideLength;
      yLoc[0] = y;
      return 1;
    }
    if (xmax > max) {
      // Create one more particles at left side
      // This is the right side
      // New Particle 1 : left side
      xLoc[0] = x - d_sideLength;
      yLoc[0] = y;
      return 1;
    }
    if (ymin < min) {
      // Create one more particles at upper side
      // This is the lower side
      // New Particle 1 : upper side
      xLoc[0] = x;
      yLoc[0] = y + d_sideLength;
      return 1;
    }
    if (ymax > max) {
      // Create one more particles at bottom side
      // This is the top side
      // New Particle 1 : bottom side
      xLoc[0] = x;
      yLoc[0] = y - d_sideLength;
      return 1;
    }
    return 0;
  }

  // Distribute circles (distribute the circular particles in a square 
  // box with the given dimensions)
  private void distributeCircles() {

    try {
      // Create a random number generator
      Random rand = new Random();
      final int MAX_ITER = 10000;

      // Pick up each particle and place in the square ..  the largest 
      // particles first
      d_sideLength = len2DCircle;
      int nofSizesCalc = d_partSizeDist.nofSizesCalc;
      for (int ii = nofSizesCalc; ii > 0; ii--) {
	int nofParts = d_partSizeDist.freq2DCalc[ii-1];
	double partDia = 0.0;
	double partDiaCurr = 0.0;
	double partDiaNext = 0.0;
	boolean fit = false;
	for (int jj = 0; jj < nofParts; jj++) {
	  
	  // Set up the particle diameter
	  partDia = d_partSizeDist.sizeCalc[ii-1];
	  partDiaCurr = partDia;
	  if (ii == 1)
	    partDiaNext = 1.0;
	  else
	    partDiaNext = d_partSizeDist.sizeCalc[ii-2]+1.0;
	  
	  // Iterate till the particle fits in the box
	  fit = false;
	  
	  int nofIter = 0;
	  double xRand, yRand, xCent, yCent, xMinPartBox, xMaxPartBox;
	  double yMinPartBox, yMaxPartBox;
	  boolean boxFit = false;
	  while (!fit) {
	    
	    // Increment the iterations and quit if the MAX_ITER is exceeded
	    if (nofIter > MAX_ITER) {
	      if (partDia <= partDiaNext) break;
	      else {
		partDia = Math.sqrt(0.9*partDia*partDia);
		nofIter = 0;
	      }
	    }
	    nofIter++;
	    
	    // Get two random numbers for the x and y
	    xRand = rand.nextDouble();
	    yRand = rand.nextDouble();

	    // Scale the co-ordinates
	    xCent = xRand*len2DCircle;
	    yCent = yRand*len2DCircle;

	    // Find if the particle fits in the box
	    xMinPartBox = xCent-partDia/2.0;
	    xMaxPartBox = xCent+partDia/2.0;
	    yMinPartBox = yCent-partDia/2.0;
	    yMaxPartBox = yCent+partDia/2.0;
	    boxFit = false;
	    if (xMinPartBox >= 0.0 && xMaxPartBox <= len2DCircle &&
		yMinPartBox >= 0.0 && yMaxPartBox <= len2DCircle) 
	      boxFit = true;

	    // Find if the particle intersects other particles already
	    // placed in the box
	    if (boxFit) {
	      int nofPartsInVector = partDiaVector.size();
	      boolean circlesIntersect = false;
	      double dia1, x1, y1;
	      for (int kk = 0; kk < nofPartsInVector; kk++) {
		dia1 = 
		  ((Double) (partDiaVector.elementAt(kk))).doubleValue();
		x1 = 
		  ((Double) (partXCentVector.elementAt(kk))).doubleValue();
		y1 = 
		  ((Double) (partYCentVector.elementAt(kk))).doubleValue();
		circlesIntersect = doCirclesIntersect(dia1, x1, y1, 
						      partDia, xCent, yCent);
		if (circlesIntersect) break;
	      } 
	      if (circlesIntersect) fit = false;
	      else {
		partDiaVector.addElement(new Double(partDia));
		partXCentVector.addElement(new Double(xCent));
		partYCentVector.addElement(new Double(yCent));
		
		// if the fit is not perfect fit the remaining volume
		// again
		if (partDiaCurr != partDia) {
		  partDia = Math.sqrt(partDiaCurr*partDiaCurr-partDia*partDia);
		  partDiaCurr = partDia;
		  partDiaNext = 0.0;
		  nofIter = 0;
		  fit = false;
		} else fit = true;
	      }
	    }
	  }
	}
      }

      int vecSize = partDiaVector.size();
      double vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	// System.out.println("Dia = "+partDiaVector.elementAt(ii)+
	// 		   " x = "+partXCentVector.elementAt(ii)+
	// 		   " y = "+partYCentVector.elementAt(ii));
	vol += 
	  Math.pow(((Double)(partDiaVector.elementAt(ii))).doubleValue(),2)
	  *Math.PI/4.0;
      }
      double volBox = Math.pow(d_sideLength,2);
      double vfrac = vol/volBox;
      // System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      // System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);

      // Fill up the rest with fines (10 microns)
      double partDia = d_partSizeDist.sizeCalc[0];
      double fracComp = d_partSizeDist.volFracInComposite/100.0;
      while (vfrac < fracComp) {

	boolean fit = false;
	int nofIter = 0;
	// System.out.println("Part Dia = "+partDia+" Vol frac = "+vfrac+
		// 	   "Vol Frac Comp = "+d_partSizeDist.volFracInComposite);
	while (!fit) {

	  // Increment the iterations and quit if the MAX_ITER is exceeded
	  if (nofIter > MAX_ITER) break;
	  nofIter++;

	  // Get two random numbers for the x and y and scale the co-ordinates
	  double xCent = rand.nextDouble()*len2DCircle;
	  double yCent = rand.nextDouble()*len2DCircle;

	  // Find if the particle fits in the box
	  double xMinPartBox = xCent-partDia/2.0;
	  double xMaxPartBox = xCent+partDia/2.0;
	  double yMinPartBox = yCent-partDia/2.0;
	  double yMaxPartBox = yCent+partDia/2.0;
	  boolean boxFit = false;
	  if (xMinPartBox >= 0.0 && xMaxPartBox <= len2DCircle &&
	      yMinPartBox >= 0.0 && yMaxPartBox <= len2DCircle) 
	    boxFit = true;

	  // Find if the particle intersects other particles already
	  // placed in the box
	  if (boxFit) {
	    int nofPartsInVector = partDiaVector.size();
	    boolean circlesIntersect = false;
	    double dia1, x1, y1;
	    for (int kk = 0; kk < nofPartsInVector; kk++) {
	      dia1 = 
		((Double) (partDiaVector.elementAt(kk))).doubleValue();
	      x1 = 
		((Double) (partXCentVector.elementAt(kk))).doubleValue();
	      y1 = 
		((Double) (partYCentVector.elementAt(kk))).doubleValue();
	      circlesIntersect = doCirclesIntersect(dia1, x1, y1, 
						    partDia, xCent, yCent);
	      if (circlesIntersect) break;
	    } 
	    if (circlesIntersect) fit = false;
	    else {
	      partDiaVector.addElement(new Double(partDia));
	      partXCentVector.addElement(new Double(xCent));
	      partYCentVector.addElement(new Double(yCent));
	      fit = true;
	    }
	  }
	}

	// Calculate the new volume
	if (fit) {
	  vfrac += Math.pow(partDia,2)*Math.PI/(4.0*volBox);
	} else {
	  partDia = Math.sqrt(0.5*partDia*partDia);
	}
      }
      vecSize = partDiaVector.size();
      vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	// System.out.println("Dia = "+partDiaVector.elementAt(ii)+
	// 		   " x = "+partXCentVector.elementAt(ii)+
	// 		   " y = "+partYCentVector.elementAt(ii));
	vol += 
	  Math.pow(((Double)(partDiaVector.elementAt(ii))).doubleValue(),2)
	  *Math.PI/4.0;
      }
      vfrac = vol/volBox;
      // System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      // System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);
    } catch (Exception e) {
      System.out.println("Some exception occured in method distributeCircles");
    }
    
    // Repaint the top canvas
    topCanvas.repaint();
    //sideCanvas.repaint();
    //frontCanvas.repaint();
  }

  // Find if circles intersect
  private boolean doCirclesIntersect(double dia1, double x1, double y1,
				     double dia2, double x2, double y2) {
    double distCent = Math.sqrt(Math.pow((x2-x1),2)+Math.pow((y2-y1),2));
    double sumRadii = dia1/2 + dia2/2;
    double gap = distCent - sumRadii;
    if (gap < 0.01*sumRadii) return true;
    //if (sumRadii > distCent) return true;
    return false;
  }

  // Distribute squares (distribute the square particles in a square 
  // box with the given dimensions)
  private void distributeSquares() {

    try {
      // Create three random number generators (two for the co-ordinates
      // of the center and one for the rotation)
      Random rand = new Random();
      final int MAX_ITER = 10000;

      // Pick up each particle and place in the square ..  the largest 
      // particles first
      d_sideLength = len2DSquare;
      int nofSizesCalc = d_partSizeDist.nofSizesCalc;
      for (int ii = nofSizesCalc; ii > 0; ii--) {
	int nofParts = d_partSizeDist.freq2DCalc[ii-1];
	double partSide = 0.0;
	double partSideCurr = 0.0;
	double partSideNext = 0.0;
	boolean fit = false;
	System.out.println("Size frac # = "+ii);
	for (int jj = 0; jj < nofParts; jj++) {
	  
	  System.out.println("Particle # = "+jj);
	  // Set up the particle side
	  partSide = d_partSizeDist.sizeCalc[ii-1];
	  partSideCurr = partSide;
	  if (ii == 1)
	    partSideNext = 1.0;
	  else
	    partSideNext = d_partSizeDist.sizeCalc[ii-2]+1.0;
	  
	  // Iterate till the particle fits in the box
	  fit = false;
	  
	  int nofIter = 0;
	  double xCent, yCent, xMinPartBox, xMaxPartBox;
	  double yMinPartBox, yMaxPartBox, rotation;
	  boolean boxFit = false;
	  double[] xCoord = new double[4];
	  double[] yCoord = new double[4];
	  Point testPt1 = new Point();
	  Point testPt2 = new Point();
	  Point testPt3 = new Point();
	  Point testPt4 = new Point();
	  while (!fit) {
	    
	    // Increment the iterations and quit if the MAX_ITER is exceeded
	    if (nofIter > MAX_ITER) {
	      if (partSide <= partSideNext) break;
	      else {
		partSide = Math.sqrt(0.9*partSide*partSide);
		nofIter = 0;
	      }
	    }
	    nofIter++;
	    
	    // Get two random numbers for the x and y and scale the co-ordinates
	    xCent = rand.nextDouble()*len2DSquare;
	    yCent = rand.nextDouble()*len2DSquare;

	    // Get the rotation amount
	    rotation = rand.nextDouble()*Math.PI/2.0; 

	    // Get the co-ordinates of the corners of the particle
	    // in counter-clockwise direction (unrotated)
	    xCoord[0] = xCent + partSide/2.0;
	    yCoord[0] = yCent + partSide/2.0;
	    xCoord[1] = xCent - partSide/2.0;
	    yCoord[1] = yCent + partSide/2.0;
	    xCoord[2] = xCent - partSide/2.0;
	    yCoord[2] = yCent - partSide/2.0;
	    xCoord[3] = xCent + partSide/2.0;
	    yCoord[3] = yCent - partSide/2.0;

	    // Rotate the co-ordinates to get the new co-ordinates
	    // and the bounding box for the square
	    xMinPartBox = xCoord[0]; 
	    xMaxPartBox = xCoord[0];
	    yMinPartBox = yCoord[0];
	    yMaxPartBox = yCoord[0];
	    for (int kk = 0; kk < 4; kk++) {
	      double xrot = xCoord[kk]*Math.cos(rotation) -
		yCoord[kk]*Math.sin(rotation);
	      double yrot = xCoord[kk]*Math.sin(rotation) +
		yCoord[kk]*Math.cos(rotation);
	      xCoord[kk] = xrot;
	      yCoord[kk] = yrot;
	      if (xMinPartBox > xrot) xMinPartBox = xrot;
	      if (xMaxPartBox < xrot) xMaxPartBox = xrot;
	      if (yMinPartBox > yrot) yMinPartBox = yrot;
	      if (yMaxPartBox < yrot) yMaxPartBox = yrot;
	    }
	    testPt1.setX(xCoord[0]); testPt1.setY(yCoord[0]);
	    testPt2.setX(xCoord[1]); testPt2.setY(yCoord[1]);
	    testPt3.setX(xCoord[2]); testPt3.setY(yCoord[2]);
	    testPt4.setX(xCoord[3]); testPt4.setY(yCoord[3]);
	    
	    // Find if the particle fits in the RVE box
	    boxFit = false;
	    if (xMinPartBox >= 0.0 && xMaxPartBox <= len2DSquare &&
		yMinPartBox >= 0.0 && yMaxPartBox <= len2DSquare) 
	      boxFit = true;

	    // Find if the particle intersects other particles already
	    // placed in the box
	    if (boxFit) {
	      int nofPartsInVector = partSideVector.size();
	      boolean squaresIntersect = false;
	      double side1;
	      try {
	      Point pt1, pt2, pt3, pt4;
	      for (int kk = 0; kk < nofPartsInVector; kk++) {
		side1 = 
		  ((Double) (partSideVector.elementAt(kk))).doubleValue();
		pt1 = (Point) (partPt1Vector.elementAt(kk));
		pt2 = (Point) (partPt2Vector.elementAt(kk));
		pt3 = (Point) (partPt3Vector.elementAt(kk));
		pt4 = (Point) (partPt4Vector.elementAt(kk));
		squaresIntersect = doSquaresIntersect(pt1, pt2, pt3, pt4,
						      testPt1, testPt2, testPt3,
						      testPt4);
		if (squaresIntersect) break;
	      } 
	      } catch (Exception e) {
		System.out.println("Exception is boxFit for loop");
	      }
	      if (squaresIntersect) fit = false;
	      else {
		partSideVector.addElement(new Double(partSide));
		partPt1Vector.addElement(new Point(testPt1));
		partPt2Vector.addElement(new Point(testPt2));
		partPt3Vector.addElement(new Point(testPt3));
		partPt4Vector.addElement(new Point(testPt4));
		partXCentVector.addElement(new Double(xCent));
		partYCentVector.addElement(new Double(yCent));
		partRotVector.addElement(new Double(rotation));
		
		// if the fit is not perfect fit the remaining volume
		// again
		if (partSideCurr != partSide) {
		  partSide = 
		    Math.sqrt(partSideCurr*partSideCurr-partSide*partSide);
		  partSideCurr = partSide;
		  partSideNext = 0.0;
		  nofIter = 0;
		  fit = false;
		} else fit = true;
	      }
	    }
	  }
	}
      }

      int vecSize = partSideVector.size();
      double vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	vol += 
	  Math.pow(((Double)(partSideVector.elementAt(ii))).doubleValue(),2);
      }
      double volBox = Math.pow(d_sideLength,2);
      double vfrac = vol/volBox;
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);

      // Fill up the rest with fines (10 microns)
      double partSide = d_partSizeDist.sizeCalc[0];
      double fracComp = d_partSizeDist.volFracInComposite/100.0;
      while (vfrac < fracComp) {

	boolean fit = false;
	int nofIter = 0;
	System.out.println("Part Side = "+partSide+" Vol frac = "+vfrac+
	 	   "Vol Frac Comp = "+fracComp);
	double xCent, yCent, xMinPartBox, xMaxPartBox;
	double yMinPartBox, yMaxPartBox, rotation;
	boolean boxFit = false;
	double[] xCoord = new double[4];
	double[] yCoord = new double[4];
	Point testPt1 = new Point();
	Point testPt2 = new Point();
	Point testPt3 = new Point();
	Point testPt4 = new Point();
	while (!fit) {

	  // Increment the iterations and quit if the MAX_ITER is exceeded
	  if (nofIter > MAX_ITER) break;
	  nofIter++;

	  // Get two random numbers for the x and y and scale the co-ordinates
	  xCent = rand.nextDouble()*len2DSquare;
	  yCent = rand.nextDouble()*len2DSquare;

	  // Get the rotation amount
	  // rotation = rand.nextDouble()*Math.PI/2.0; 

	  // Get the co-ordinates of the corners of the particle
	  // in counter-clockwise direction (unrotated)
	  xCoord[0] = xCent + partSide/2.0;
	  yCoord[0] = yCent + partSide/2.0;
	  xCoord[1] = xCent - partSide/2.0;
	  yCoord[1] = yCent + partSide/2.0;
	  xCoord[2] = xCent - partSide/2.0;
	  yCoord[2] = yCent - partSide/2.0;
	  xCoord[3] = xCent + partSide/2.0;
	  yCoord[3] = yCent - partSide/2.0;

	  // Rotate the co-ordinates to get the new co-ordinates
	  // and the bounding box for the square
	  xMinPartBox = xCoord[0]; 
	  xMaxPartBox = xCoord[0];
	  yMinPartBox = yCoord[0];
	  yMaxPartBox = yCoord[0];
	  for (int ii = 1; ii < 4; ii++) {
	    // double xrot = xCoord[ii]*Math.cos(rotation) -
	    //   yCoord[ii]*Math.sin(rotation);
	    // double yrot = xCoord[ii]*Math.sin(rotation) +
	    //   yCoord[ii]*Math.cos(rotation);
	    // xCoord[ii] = xrot;
	    // yCoord[ii] = yrot;
	    double xrot = xCoord[ii];
	    double yrot = yCoord[ii];
	    if (xMinPartBox > xrot) xMinPartBox = xrot;
	    if (xMaxPartBox < xrot) xMaxPartBox = xrot;
	    if (yMinPartBox > yrot) yMinPartBox = yrot;
	    if (yMaxPartBox < yrot) yMaxPartBox = yrot;
	  }

	  testPt1.setX(xCoord[0]); testPt1.setY(yCoord[0]);
	  testPt2.setX(xCoord[1]); testPt2.setY(yCoord[1]);
	  testPt3.setX(xCoord[2]); testPt3.setY(yCoord[2]);
	  testPt4.setX(xCoord[3]); testPt4.setY(yCoord[3]);

	  boxFit = false;
	  if (xMinPartBox >= 0.0 && xMaxPartBox <= len2DSquare &&
	      yMinPartBox >= 0.0 && yMaxPartBox <= len2DSquare) 
	    boxFit = true;

	  // Find if the particle intersects other particles already
	  // placed in the box
	  if (boxFit) {
	    int nofPartsInVector = partSideVector.size();
	    boolean squaresIntersect = false;
	    double side1, x1, y1;
	    Point pt1, pt2, pt3, pt4;
	    for (int kk = 0; kk < nofPartsInVector; kk++) {
	      side1 = 
		((Double) (partSideVector.elementAt(kk))).doubleValue();
	      pt1 = (Point) (partPt1Vector.elementAt(kk));
	      pt2 = (Point) (partPt2Vector.elementAt(kk));
	      pt3 = (Point) (partPt3Vector.elementAt(kk));
	      pt4 = (Point) (partPt4Vector.elementAt(kk));
	      squaresIntersect = doSquaresIntersect(pt1, pt2, pt3, pt4,
						    testPt1, testPt2, testPt3,
						    testPt4);
	      if (squaresIntersect) break;
	    } 
	    if (squaresIntersect) fit = false;
	    else {
	      partSideVector.addElement(new Double(partSide));
	      partPt1Vector.addElement(new Point(testPt1));
	      partPt2Vector.addElement(new Point(testPt2));
	      partPt3Vector.addElement(new Point(testPt3));
	      partPt4Vector.addElement(new Point(testPt4));
	      partXCentVector.addElement(new Double(xCent));
	      partYCentVector.addElement(new Double(yCent));
	      partRotVector.addElement(new Double(0.0));
	      fit = true;
	    }
	  }
	}

	// Calculate the new volume
	if (fit) {
	  vfrac += Math.pow(partSide,2)/volBox;
	} else {
	  partSide = Math.sqrt(0.5*partSide*partSide);
	}
      }
      vecSize = partSideVector.size();
      vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	// System.out.println("Side = "+partSideVector.elementAt(ii)+
	// 		   " x = "+partXCentVector.elementAt(ii)+
	// 		   " y = "+partYCentVector.elementAt(ii));
	vol += 
	  Math.pow(((Double)(partSideVector.elementAt(ii))).doubleValue(),2);
      }
      vfrac = vol/volBox;
      // System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      // System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);
      // Calculate the weight fraction of the various components based on 
      // original size distribution
    } catch (Exception e) {
      System.out.println("Some exception occured in method distributeSquare");
    }
    
    // Repaint the top canvas
    topCanvas.repaint();
    //sideCanvas.repaint();
    //frontCanvas.repaint();
  }

  // Find if squares intersect
  private boolean doSquaresIntersect(Point pt1, Point pt2, Point pt3, Point pt4,
				     Point pt11, Point pt21, Point pt31,
				     Point pt41) {
    boolean intersect = false;
    // First find if the first set of pts lies in the second square
    intersect = inSquare(pt1, pt11, pt21, pt31, pt41);
    if (intersect) return true;
    intersect = inSquare(pt2, pt11, pt21, pt31, pt41);
    if (intersect) return true;
    intersect = inSquare(pt3, pt11, pt21, pt31, pt41);
    if (intersect) return true;
    intersect = inSquare(pt4, pt11, pt21, pt31, pt41);
    if (intersect) return true;
    // First find if the second set of pts lies in the first square
    intersect = inSquare(pt11, pt1, pt2, pt3, pt4);
    if (intersect) return true;
    intersect = inSquare(pt21, pt1, pt2, pt3, pt4);
    if (intersect) return true;
    intersect = inSquare(pt31, pt1, pt2, pt3, pt4);
    if (intersect) return true;
    intersect = inSquare(pt41, pt1, pt2, pt3, pt4);
    if (intersect) return true;
    // Find edge intersections
    intersect = edgesIntersect(pt1, pt2, pt11, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt1, pt2, pt21, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt1, pt2, pt31, pt41);
    if (intersect) return true;
    intersect = edgesIntersect(pt1, pt2, pt41, pt11);
    if (intersect) return true;
    intersect = edgesIntersect(pt2, pt3, pt11, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt2, pt3, pt21, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt2, pt3, pt31, pt41);
    if (intersect) return true;
    intersect = edgesIntersect(pt2, pt3, pt41, pt11);
    if (intersect) return true;
    intersect = edgesIntersect(pt3, pt4, pt11, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt3, pt4, pt21, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt3, pt4, pt31, pt41);
    if (intersect) return true;
    intersect = edgesIntersect(pt3, pt4, pt41, pt11);
    if (intersect) return true;
    intersect = edgesIntersect(pt4, pt1, pt11, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt4, pt1, pt21, pt21);
    if (intersect) return true;
    intersect = edgesIntersect(pt4, pt1, pt31, pt41);
    if (intersect) return true;
    intersect = edgesIntersect(pt4, pt1, pt41, pt11);
    if (intersect) return true;
    return false;
  }

  // Find if a point is within a square
  private boolean inSquare(Point pt, Point sqrPt1, Point sqrPt2, Point sqrPt3,
			   Point sqrPt4) {
    double x = pt.getX();
    double y = pt.getY();
    double[] xsqr = new double[4];
    double[] ysqr = new double[4];
    xsqr[0] = sqrPt1.getX(); ysqr[0] = sqrPt1.getY();
    xsqr[1] = sqrPt2.getX(); ysqr[1] = sqrPt2.getY();
    xsqr[2] = sqrPt3.getX(); ysqr[2] = sqrPt3.getY();
    xsqr[3] = sqrPt4.getX(); ysqr[3] = sqrPt4.getY();
    boolean c = false;

    for (int ii = 0, jj = 3; ii < 4; jj = ii++) {
      double xpi = xsqr[ii];
      double xpj = xsqr[jj];
      double ypi = ysqr[ii];
      double ypj = ysqr[jj];
      if ((((ypi <= y) && (y < ypj)) ||
	   ((ypj <= y) && (y < ypi))) &&
	  (x < (xpj - xpi) * ( y - ypi) / (ypj - ypi) + xpi)) {
	c = !c;
      }
    }
    return c;
  }

  // Find if edges intersect
  private boolean edgesIntersect(Point ptA, Point ptB, Point ptC, Point ptD) {
    double xa = ptA.getX();
    double xb = ptB.getX();
    double xc = ptC.getX();
    double xd = ptD.getX();
    double ya = ptA.getY();
    double yb = ptB.getY();
    double yc = ptC.getY();
    double yd = ptD.getY();
    double den = (xb-xa)*(yd-yc) - (yb-ya)*(xd-xc);
    double numr = (ya-yc)*(xd-xc) - (xa-xc)*(yd-yc);
    double nums = (ya-yc)*(xb-xa) - (xa-xc)*(yb-ya);
    if (den == 0.0) return false; // Edges are parallel
    double rVal = numr/den;
    double sVal = nums/den;
    if (rVal < 0.0 || rVal > 1.0 || sVal < 0.0 || sVal > 1.0) return false;
    return true;
  }

  // Distribute spheres (distribute the spherical particles in a cube 
  // box with the given dimensions)
  private void distributeSpheres() {

    try {
      // Create a random number generator for the center co-ordinates
      Random rand = new Random();
      final int MAX_ITER = 30000;

      // Pick up each particle and place in the cube ..  the largest 
      // particles first
      d_sideLength = len3DSphere;
      int nofSizesCalc = d_partSizeDist.nofSizesCalc;
      for (int ii = nofSizesCalc; ii > 0; ii--) {
	int nofParts = d_partSizeDist.freq3DCalc[ii-1];
	double partDia = 0.0;
	double partDiaCurr = 0.0;
	double partDiaNext = 0.0;
	boolean fit = false;
	System.out.println("Particle size fraction # = "+ii);
	for (int jj = 0; jj < nofParts; jj++) {
	  
	  // Set up the particle diameter
	  System.out.println("Particle # = "+jj);
	  partDia = d_partSizeDist.sizeCalc[ii-1];
	  partDiaCurr = partDia;
	  if (ii == 1)
	    partDiaNext = 1.0;
	  else
	    partDiaNext = d_partSizeDist.sizeCalc[ii-2]+1.0;
	  
	  // Iterate till the particle fits in the box
	  fit = false;
	  
	  int nofIter = 0;
	  double xCent, yCent, zCent, xMinPartBox, xMaxPartBox;
	  double yMinPartBox, yMaxPartBox, zMinPartBox, zMaxPartBox;
	  boolean boxFit = false;
	  while (!fit) {
	    
	    // Increment the iterations and quit if the MAX_ITER is exceeded
	    if (nofIter > MAX_ITER) {
	      //if (partDia < 5.0) break;
	      //else {
	      //partDia = Math.pow((0.9*Math.pow(partDia,3.0)),(1.0/3.0));
	      //nofIter = 0;
	      //}
	      len3DSphere *= 1.0005;
	      if (partDiaVector.size() > 0) {
		partDiaVector.removeAllElements();
		partXCentVector.removeAllElements();
		partYCentVector.removeAllElements();
		partZCentVector.removeAllElements();
	      }
		
	      distributeSpheres();
	      return;
	      
	    }
	    nofIter++;
	    
	    // Get three random numbers for the x,y and z and scale
	    xCent = rand.nextDouble()*len3DSphere;
	    yCent = rand.nextDouble()*len3DSphere;
	    zCent = rand.nextDouble()*len3DSphere;

	    // Find if the particle fits in the box
	    xMinPartBox = xCent-partDia/2.0;
	    xMaxPartBox = xCent+partDia/2.0;
	    yMinPartBox = yCent-partDia/2.0;
	    yMaxPartBox = yCent+partDia/2.0;
	    zMinPartBox = zCent-partDia/2.0;
	    zMaxPartBox = zCent+partDia/2.0;
	    boxFit = false;
	    if (xMinPartBox >= 0.0 && xMaxPartBox <= len3DSphere &&
		yMinPartBox >= 0.0 && yMaxPartBox <= len3DSphere &&
		zMinPartBox >= 0.0 && zMaxPartBox <= len3DSphere) 
	      boxFit = true;

	    // Find if the particle intersects other particles already
	    // placed in the box
	    if (boxFit) {
	      int nofPartsInVector = partDiaVector.size();
	      boolean spheresIntersect = false;
	      double dia1, x1, y1, z1;
	      for (int kk = 0; kk < nofPartsInVector; kk++) {
		dia1 = 
		  ((Double) (partDiaVector.elementAt(kk))).doubleValue();
		x1 = 
		  ((Double) (partXCentVector.elementAt(kk))).doubleValue();
		y1 = 
		  ((Double) (partYCentVector.elementAt(kk))).doubleValue();
		z1 = 
		  ((Double) (partZCentVector.elementAt(kk))).doubleValue();
		spheresIntersect = doSpheresIntersect(dia1, x1, y1, z1,
					      partDia, xCent, yCent, zCent);
		if (spheresIntersect) break;
	      } 
	      if (spheresIntersect) fit = false;
	      else {
		partDiaVector.addElement(new Double(partDia));
		partXCentVector.addElement(new Double(xCent));
		partYCentVector.addElement(new Double(yCent));
		partZCentVector.addElement(new Double(zCent));
		
		System.out.println("Dia = "+partDia+ " x = "+xCent+ " y = "+yCent+ " z = "+zCent);
		// if the fit is not perfect fit the remaining volume
		// again
		if (partDiaCurr != partDia) {
		  partDia = 
		    Math.pow(Math.pow(partDiaCurr,3)-Math.pow(partDia,3),(1.0/3.0));
		  partDiaCurr = partDia;
		  partDiaNext = 0.0;
		  nofIter = 0;
		  fit = false;
		} else {
		  fit = true;
		}
	      }
	    }
	  }
	}
	topCanvas.paintImmediately();
      }

      // calculate the volume of the particles
      int vecSize = partDiaVector.size();
      double vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	System.out.println("Dia = "+partDiaVector.elementAt(ii)+
	" x = "+partXCentVector.elementAt(ii)+
	" y = "+partYCentVector.elementAt(ii)+
	" z = "+partZCentVector.elementAt(ii));
	vol += 
	  Math.pow(((Double)(partDiaVector.elementAt(ii))).doubleValue(),3)
	  *Math.PI/6.0;
      }
      double volBox = Math.pow(d_sideLength,3);
      double vfrac = vol/volBox;
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);

      // Carry out MD simulation to redistribute the particles
      // MDSimulation mdSim = new MDSimulation();
      // mdSim.fixedPartSizeMD();
      // mdSim.varPartSizeMD();

      // Fill up the rest with fines 
      int currentPartSize = d_partSizeDist.nofSizesCalc-1;
      double partDia = d_partSizeDist.sizeCalc[currentPartSize];
      double fracComp = d_partSizeDist.volFracInComposite/100.0;
      // TEMPORARY PATCH
      while (vfrac < fracComp && partDia > 10.0) {

	boolean fit = false;
	int nofIter = 0;
	System.out.println("Part Dia = "+partDia+" Vol frac = "+vfrac+
			   "Vol Frac Comp = "+d_partSizeDist.volFracInComposite);
	while (!fit) {

	  // Increment the iterations and quit if the MAX_ITER is exceeded
	  if (nofIter > MAX_ITER) break;
	  nofIter++;

	  // Get two random numbers for the x and y and scale the co-ordinates
	  double xCent = rand.nextDouble()*len3DSphere;
	  double yCent = rand.nextDouble()*len3DSphere;
	  double zCent = rand.nextDouble()*len3DSphere;

	  // Find if the particle fits in the box
	  double xMinPartBox = xCent-partDia/2.0;
	  double xMaxPartBox = xCent+partDia/2.0;
	  double yMinPartBox = yCent-partDia/2.0;
	  double yMaxPartBox = yCent+partDia/2.0;
	  double zMinPartBox = zCent-partDia/2.0;
	  double zMaxPartBox = zCent+partDia/2.0;
	  boolean boxFit = false;
	  if (xMinPartBox >= 0.0 && xMaxPartBox <= len3DSphere &&
	      yMinPartBox >= 0.0 && yMaxPartBox <= len3DSphere && 
	      zMinPartBox >= 0.0 && zMaxPartBox <= len3DSphere) 
	    boxFit = true;

	  // Find if the particle intersects other particles already
	  // placed in the box
	  if (boxFit) {
	    int nofPartsInVector = partDiaVector.size();
	    boolean spheresIntersect = false;
	    double dia1, x1, y1, z1;
	    for (int kk = 0; kk < nofPartsInVector; kk++) {
	      dia1 = 
		((Double) (partDiaVector.elementAt(kk))).doubleValue();
	      x1 = 
		((Double) (partXCentVector.elementAt(kk))).doubleValue();
	      y1 = 
		((Double) (partYCentVector.elementAt(kk))).doubleValue();
	      z1 = 
		((Double) (partZCentVector.elementAt(kk))).doubleValue();
	      spheresIntersect = doSpheresIntersect(dia1, x1, y1, z1,
					    partDia, xCent, yCent, zCent);
	      if (spheresIntersect) break;
	    } 
	    if (spheresIntersect) fit = false;
	    else {
	      partDiaVector.addElement(new Double(partDia));
	      partXCentVector.addElement(new Double(xCent));
	      partYCentVector.addElement(new Double(yCent));
	      partZCentVector.addElement(new Double(zCent));
	      fit = true;
	    }
	  }
	}

	// Calculate the new volume
	if (fit) {
	  vfrac += Math.pow(partDia,3)*Math.PI/(6.0*volBox);
	} else {
	  currentPartSize--;
	  if (currentPartSize < 1) 
	    partDia = Math.pow(0.9*Math.pow(partDia,3),(1.0/3.0));
	  else
	    partDia = d_partSizeDist.sizeCalc[currentPartSize];
	}
      }
      vecSize = partDiaVector.size();
      vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	System.out.println("Dia = "+partDiaVector.elementAt(ii)+
	 		   " x = "+partXCentVector.elementAt(ii)+
	 		   " y = "+partYCentVector.elementAt(ii));
	vol += 
	  Math.pow(((Double)(partDiaVector.elementAt(ii))).doubleValue(),3)
	  *Math.PI/6.0;
      }
      vfrac = vol/volBox;
      System.out.println("Final values");
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);
    } catch (Exception e) {
      System.out.println("Some exception occured in method distributeSpheres");
    }
    
    // Repaint the top canvas
    topCanvas.repaint();
    //sideCanvas.repaint();
    //frontCanvas.repaint();
  }

  // Find if spheres intersect
  private boolean doSpheresIntersect(double dia1, double x1, double y1,
				     double z1, double dia2, double x2, 
				     double y2, double z2) {
    double distCent = 
      Math.sqrt(Math.pow((x2-x1),2)+Math.pow((y2-y1),2)+Math.pow((z2-z1),2));
    double sumRadii = dia1/2 + dia2/2;
    if (sumRadii > distCent) return true;
    return false;
  }

  //**************************************************************************
  // Class   : MDSimulation
  // Purpose : Carries out a Molecular Dynamics Simulation with fixed 
  //           particle sizes and then with increasing particle sizes.
  //           for spherical particles.
  //**************************************************************************
  private class MDSimulation {

    // private data
    private int nofParts = 0;
    private double[] xVel = null;
    private double[] yVel = null;
    private double[] zVel = null;
    private double[] xCent = null;
    private double[] yCent = null;
    private double[] zCent = null;
    private double[] dia = null;
    private double[] collisionTime = null;
    private int[] partner = null;
    
    // Constants
    final double TIME_LONG = 1.0e12;
    final int MAX_COLL = 20000;

    // Constructor (assumes particle details are already available)
    public MDSimulation() {
      nofParts = partDiaVector.size();
      xVel = new double[nofParts];
      yVel = new double[nofParts];
      zVel = new double[nofParts];
      xCent = new double[nofParts];
      yCent = new double[nofParts];
      zCent = new double[nofParts];
      dia = new double[nofParts];
      collisionTime = new double[nofParts];
      partner = new int[nofParts];
      initialize();
    }

    // Initialization method
    private void initialize() {
      
      // 1)Assign random velocities to the particles.  The value of the
      // velocity vector varies from 0.0 to 1.0. 
      // 2) Store the center and diameter of the particle.
      // 3) Initialize the collision lists
      Random rand = new Random();

      for (int ii = 0; ii < nofParts; ii++) {
	// 1) Velocity
	double sign = 1.0;
	if (rand.nextDouble() < 0.5) sign = -1.0;
	xVel[ii] = sign*rand.nextDouble();
	yVel[ii] = sign*rand.nextDouble();
	zVel[ii] = sign*rand.nextDouble();
	// 2) Center and Diameter
	xCent[ii] = ((Double) partXCentVector.elementAt(ii)).doubleValue();
	yCent[ii] = ((Double) partYCentVector.elementAt(ii)).doubleValue();
	zCent[ii] = ((Double) partZCentVector.elementAt(ii)).doubleValue();
	dia[ii] = ((Double) partDiaVector.elementAt(ii)).doubleValue();
	// 3) Initialize the collision lists
	collisionTime[ii] = TIME_LONG;
	partner[ii] = nofParts;
      }
    }

    // Method to compute the first run of the Molecular Dynamics simulation
    // of hard particles with particle sizes fixed
    private void fixedPartSizeMD() {

      // Check for particle pair overlaps and calculate energy
      boolean overlap = checkOverlap();
      if (overlap) 
	System.out.println("Particle Overlap in initial configuration");

      // set up initial collision lists
      for (int ii = 0; ii < nofParts-1; ii++) upCollisionLists(ii);

      // Start time stepping
      double tt = 0.0;
      int ii = 0, jj = 0;
      for (int coll = 0; coll < MAX_COLL; coll++) {

	// Locate minimum collision time
	double tij = TIME_LONG;
	for (int kk = 0; kk < nofParts; kk++) {
	  if (collisionTime[kk] < tij) {
	    tij = collisionTime[kk];
	    ii = kk;
	  }
	}
	jj = partner[ii];

	// Move particles forward by time tij, reduce collision times,
	// apply periodic boundaries
	tt += tij;
	for (int kk = 0; kk < nofParts; kk++) {
	  collisionTime[kk] -= tij;
	  xCent[kk] += (xVel[kk]*tij);
	  yCent[kk] += (yVel[kk]*tij);
	  zCent[kk] += (zVel[kk]*tij);
	}

	// Compute collision dynamics
	bump(ii,jj);

	// Reset the collision lists for those particles which need it
	for (int kk = 0; kk < nofParts; kk++) {
	  if ((kk == ii) || (partner[kk] == ii) || (kk == jj) ||
			  (partner[kk] == jj)) upCollisionLists(kk);
	}
	downCollisionLists(ii);
	downCollisionLists(jj);
      }

      // check for overlaps
      overlap = checkOverlap();
      if (overlap) {
	System.out.println("Particle overlap in final configuration");
      }

      // Update the locations of the particles
      updateParticles();
    }

    // Method to compute the second run of the Molecular Dynamics simulation
    // of hard particles with particle sizes increasing steadily
    private void varPartSizeMD() {

      // set up the rate of volume increase
      double rate = 1.0;
      
      // set up the new volume/dia
      double tt = 0.01;
      for (int ii = 0; ii < nofParts; ii++) updateDia(ii, rate, tt);

      // Check for particle pair overlaps
      boolean overlap = checkOverlap();
      if (overlap) 
	System.out.println("Particle Overlap in initial configuration");

      // set up initial collision lists
      for (int ii = 0; ii < nofParts-1; ii++) upCollisionLists(ii);

      // Start time stepping
      int ii = 0, jj = 0;
      for (int coll = 0; coll < MAX_COLL; coll++) {

	// Locate minimum collision time
	double tij = TIME_LONG;
	for (int kk = 0; kk < nofParts; kk++) {
	  if (collisionTime[kk] < tij) {
	    tij = collisionTime[kk];
	    ii = kk;
	  }
	}
	jj = partner[ii];

	// Move particles forward by time tij, reduce collision times,
	// apply periodic boundaries
	tt += tij;
	for (int kk = 0; kk < nofParts; kk++) {
	  collisionTime[kk] -= tij;
	  xCent[kk] += (xVel[kk]*tij);
	  yCent[kk] += (yVel[kk]*tij);
	  zCent[kk] += (zVel[kk]*tij);
	}

	// Compute collision dynamics
	bump(ii,jj);

	// Reset the collision lists for those particles which need it
	for (int kk = 0; kk < nofParts; kk++) {
	  if ((kk == ii) || (partner[kk] == ii) || (kk == jj) ||
			  (partner[kk] == jj)) upCollisionLists(kk);
	}
	downCollisionLists(ii);
	downCollisionLists(jj);

	// Update the diamters again
	for (int kk = 0; kk < nofParts; kk++) updateDia(kk, rate, tt);
      }

      // check for overlaps
      overlap = checkOverlap();
      if (overlap) {
	System.out.println("Particle overlap in final configuration");
      }

      // Update the locations of the particles
      updateParticles();

      // Restore the diameters of the particles
      // for (int kk = 0; kk < nofParts; kk++) restoreDia(kk, rate, tt);
    }

    // Method to update the volume based on the rate of increase
    private void updateDia(int ii, double rate, double tt) {
      System.out.print("Old Dia = "+dia[ii]+" tt = "+tt);
      dia[ii] = dia[ii]*(1.0+rate*tt);
      System.out.println(" New Dia = "+dia[ii]);
    }

    // Method to restore the volume based on the rate of increase
    private void restoreDia(int ii, double rate, double tt) {
      dia[ii] = Math.pow(Math.pow(dia[ii],3)/(1.0-rate*tt),(1.0/3.0));
    }

    // Method to check for particle pair overlap
    private boolean checkOverlap() {

      boolean overlap = false;

      // Loop thru the particles
      for (int ii = 0; ii < nofParts-1; ii++) {
	double rxi = xCent[ii];
	double ryi = yCent[ii];
	double rzi = zCent[ii];
	double diai = dia[ii];
	for (int jj = ii+1; jj < nofParts; jj++) {
	  double rxij = rxi - xCent[jj];
	  double ryij = ryi - yCent[jj];
	  double rzij = rzi - zCent[jj];
	  double diaj = dia[jj];
	  double distCent = 
	    Math.sqrt(Math.pow(rxij,2.0)+Math.pow(ryij,2.0)+Math.pow(rzij,2.0));
	  double sumRad = (diai+diaj)/2.0;
	  if (distCent < sumRad) overlap = true;
	}
      }
      return overlap;
    }

    // Method to update the collision lists .. looks for collisions with
    // particles jj > ii
    private void upCollisionLists(int ii) {

      // Loops thru the particles
      for (int jj = ii+1; jj < nofParts; jj++) {
	double rxij = xCent[ii] - xCent[jj];
	double ryij = yCent[ii] - yCent[jj];
	double rzij = zCent[ii] - zCent[jj];
	double vxij = xVel[ii] - xVel[jj];
	double vyij = yVel[ii] - yVel[jj];
	double vzij = zVel[ii] - zVel[jj];
	double bij = rxij*vxij + ryij*vyij + rzij*vzij;
	double sigma = dia[ii]/2.0+dia[jj]/2.0;
	double sigsq = Math.pow(sigma,2.0);
	if (bij < 0.0) {
	  double rijsq = Math.pow(rxij,2)+Math.pow(ryij,2)+Math.pow(rzij,2);
	  double vijsq = Math.pow(vxij,2)+Math.pow(vyij,2)+Math.pow(vzij,2);
	  double discr = Math.pow(bij,2)-vijsq*(rijsq-sigsq);
	  if (discr > 0.0) {
	    double tij = (-bij - Math.sqrt(discr))/vijsq;
	    if (tij < collisionTime[ii]) {
	      collisionTime[ii] = tij;
	      partner[ii] = jj;
	    }
	  }
	}
      }
    } // End of up collision lists

    // Method to update the collision lists .. looks for collisions with
    // particles jj > ii
    private void downCollisionLists(int jj) {

      // Loops thru the particles
      for (int ii = 0; ii < jj-1; ii++) {
	double rxij = xCent[ii] - xCent[jj];
	double ryij = yCent[ii] - yCent[jj];
	double rzij = zCent[ii] - zCent[jj];
	double vxij = xVel[ii] - xVel[jj];
	double vyij = yVel[ii] - yVel[jj];
	double vzij = zVel[ii] - zVel[jj];
	double bij = rxij*vxij + ryij*vyij + rzij*vzij;
	double sigma = dia[ii]/2.0+dia[jj]/2.0;
	double sigsq = Math.pow(sigma,2.0);
	if (bij < 0.0) {
	  double rijsq = Math.pow(rxij,2)+Math.pow(ryij,2)+Math.pow(rzij,2);
	  double vijsq = Math.pow(vxij,2)+Math.pow(vyij,2)+Math.pow(vzij,2);
	  double discr = Math.pow(bij,2)-vijsq*(rijsq-sigsq);
	  if (discr > 0.0) {
	    double tij = (-bij - Math.sqrt(discr))/vijsq;
	    if (tij < collisionTime[ii]) {
	      collisionTime[ii] = tij;
	      partner[ii] = jj;
	    }
	  }
	}
      }
    } // End of down collision lists

    // Method to compute collision dynamics
    private void bump(int ii, int jj) {
      double rxij = xCent[ii] - xCent[jj];
      double ryij = yCent[ii] - yCent[jj];
      double rzij = zCent[ii] - zCent[jj];
      double vxij = xVel[ii] - xVel[jj];
      double vyij = yVel[ii] - yVel[jj];
      double vzij = zVel[ii] - zVel[jj];
      double sigma = dia[ii]/2.0+dia[jj]/2.0;
      double sigsq = Math.pow(sigma,2.0);
      double factor = (rxij*vxij+ryij*vyij+rzij*vzij)/sigsq;
      double delvx = -factor*rxij;
      double delvy = -factor*ryij;
      double delvz = -factor*rzij;
      xVel[ii] += delvx;
      xVel[jj] -= delvx;
      yVel[ii] += delvy;
      yVel[jj] -= delvy;
      zVel[ii] += delvz;
      zVel[jj] -= delvz;
    }

    // Update the locations and diameters of the particles
    private void updateParticles() {
      if (nofParts > 0) {
	partDiaVector.removeAllElements();
	partXCentVector.removeAllElements();
	partYCentVector.removeAllElements();
	partZCentVector.removeAllElements();
	for (int ii = 0; ii < nofParts; ii++) {
	  partDiaVector.addElement(new Double(dia[ii]));
	  partXCentVector.addElement(new Double(xCent[ii]));
	  partYCentVector.addElement(new Double(yCent[ii]));
	  partZCentVector.addElement(new Double(zCent[ii]));
	}
      }
    }

  }

  // Distribute cubes (distribute the cubical particles in a cubical 
  // box with the given dimensions)
  private void distributeCubes() {

    try {
      // Create three random number generators (for the co-ordinates
      // of the center)
      Random rand = new Random();
      final int MAX_ITER = 20000;

      // Pick up each particle and place in the cube ..  the largest 
      // particles first
      d_sideLength = len3DCube;
      int nofSizesCalc = d_partSizeDist.nofSizesCalc;
      for (int ii = nofSizesCalc; ii > 0; ii--) {
	int nofParts = d_partSizeDist.freq3DCalc[ii-1];
	double partSide = 0.0;
	double partSideCurr = 0.0;
	double partSideNext = 0.0;
	boolean fit = false;
	System.out.println("Size frac # = "+ii);
	for (int jj = 0; jj < nofParts; jj++) {
	  
	  System.out.println("Particle # = "+jj);
	  // Set up the particle side
	  partSide = d_partSizeDist.sizeCalc[ii-1];
	  partSideCurr = partSide;
	  if (ii == 1)
	    partSideNext = 1.0;
	  else
	    partSideNext = d_partSizeDist.sizeCalc[ii-2]+1.0;
	  
	  // Iterate till the particle fits in the box
	  fit = false;
	  
	  int nofIter = 0;
	  double xCent, yCent, zCent;
	  boolean boxFit = false;
	  while (!fit) {
	    
	    // Increment the iterations and quit if the MAX_ITER is exceeded
	    if (nofIter > MAX_ITER) {
	      if (partSide <= partSideNext) break;
	      else {
		partSide = Math.pow((0.9*Math.pow(partSide,3)),(1.0/3.0));
		nofIter = 0;
	      }
	    }
	    nofIter++;
	    
	    // Get two random numbers for the x,y,z and scale the co-ordinates
	    xCent = rand.nextDouble()*len3DCube;
	    yCent = rand.nextDouble()*len3DCube;
	    zCent = rand.nextDouble()*len3DCube;

	    // Find if the particle fits in the RVE box
	    double centCube = len3DCube/2.0;
	    boxFit = isCube2InCube1(len3DCube, centCube, centCube, centCube,
				      partSide, xCent, yCent, zCent);

	    // Find if the particle intersects other particles already
	    // placed in the box
	    if (boxFit) {
	      int nofPartsInVector = partSideVector.size();
	      boolean cubesIntersect = false;
	      double side, xx, yy, zz;
	      try {
		for (int kk = 0; kk < nofPartsInVector; kk++) {
		  side = 
		    ((Double) (partSideVector.elementAt(kk))).doubleValue();
		  xx = 
		    ((Double) (partXCentVector.elementAt(kk))).doubleValue();
		  yy = 
		    ((Double) (partYCentVector.elementAt(kk))).doubleValue();
		  zz = 
		    ((Double) (partZCentVector.elementAt(kk))).doubleValue();
		  cubesIntersect = doCubesIntersect(side, xx, yy, zz, partSide, 
						    xCent, yCent, zCent);
		  if (cubesIntersect) break;
		} 
	      } catch (Exception e) {
		System.out.println("Exception in boxFit for loop");
	      }
	      if (cubesIntersect) fit = false;
	      else {
		partSideVector.addElement(new Double(partSide));
		partXCentVector.addElement(new Double(xCent));
		partYCentVector.addElement(new Double(yCent));
		partZCentVector.addElement(new Double(zCent));
		
		// if the fit is not perfect fit the remaining volume
		// again
		if (partSideCurr != partSide) {
		  partSide = 
		    Math.pow((Math.pow(partSideCurr,3)-Math.pow(partSide,3)),
			     (1.0/3.0));
		  partSideCurr = partSide;
		  partSideNext = 0.0;
		  nofIter = 0;
		  fit = false;
		} else fit = true;
	      }
	    }
	  }
	}
      }

      int vecSize = partSideVector.size();
      double vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	vol += 
	  Math.pow(((Double)(partSideVector.elementAt(ii))).doubleValue(),3);
      }
      double volBox = Math.pow(d_sideLength,3);
      double vfrac = vol/volBox;
      System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);

      // Fill up the rest with fines (10 microns)
      double partSide = d_partSizeDist.sizeCalc[0];
      double fracComp = d_partSizeDist.volFracInComposite/100.0;
      while (vfrac < fracComp) {

	boolean fit = false;
	int nofIter = 0;
	System.out.println("Part Side = "+partSide+" Vol frac = "+vfrac+
	 	   "Vol Frac Comp = "+fracComp);
	double xCent, yCent, zCent;
	boolean boxFit = false;
	while (!fit) {

	  // Increment the iterations and quit if the MAX_ITER is exceeded
	  if (nofIter > MAX_ITER) break;
	  nofIter++;

	  // Get three random numbers for the x,y,z and scale the co-ordinates
	  xCent = rand.nextDouble()*len3DCube;
	  yCent = rand.nextDouble()*len3DCube;
	  zCent = rand.nextDouble()*len3DCube;

	  // See if the particle fits in the bounding box
	  double centCube = len3DCube/2.0;
	  boxFit = isCube2InCube1(len3DCube, centCube, centCube, centCube,
				  partSide, xCent, yCent, zCent);

	  // Find if the particle intersects other particles already
	  // placed in the box
	  if (boxFit) {
	    int nofPartsInVector = partSideVector.size();
	    boolean cubesIntersect = false;
	    double side, xx, yy, zz;
	    for (int kk = 0; kk < nofPartsInVector; kk++) {
	      side = 
		((Double) (partSideVector.elementAt(kk))).doubleValue();
	      xx = 
		((Double) (partXCentVector.elementAt(kk))).doubleValue();
	      yy = 
		((Double) (partYCentVector.elementAt(kk))).doubleValue();
	      zz = 
		((Double) (partZCentVector.elementAt(kk))).doubleValue();
	      cubesIntersect = doCubesIntersect(side, xx, yy, zz, partSide, 
						xCent, yCent, zCent);
	      if (cubesIntersect) break;
	    } 
	    if (cubesIntersect) fit = false;
	    else {
	      partSideVector.addElement(new Double(partSide));
	      partXCentVector.addElement(new Double(xCent));
	      partYCentVector.addElement(new Double(yCent));
	      partZCentVector.addElement(new Double(zCent));
	      fit = true;
	    }
	  }
	}

	// Calculate the new volume
	if (fit) {
	  vfrac += Math.pow(partSide,3)/volBox;
	} else {
	  partSide = Math.pow((0.5*Math.pow(partSide,3)),(1.0/3.0));
	}
      }
      vecSize = partSideVector.size();
      vol = 0.0;
      for (int ii = 0; ii < vecSize; ii++) {
	System.out.println("Side = "+partSideVector.elementAt(ii)+
	" x = "+partXCentVector.elementAt(ii)+
	" y = "+partYCentVector.elementAt(ii)+
	" z = "+partZCentVector.elementAt(ii));
	vol += 
	  Math.pow(((Double)(partSideVector.elementAt(ii))).doubleValue(),3);
      }
      vfrac = vol/volBox;
      // System.out.println("No of parts = "+vecSize+" Vol frac = "+(vol/volBox));
      // System.out.println("Volume of parts = "+vol+" Box vol = "+volBox);
      // Calculate the weight fraction of the various components based on 
      // original size distribution
    } catch (Exception e) {
      System.out.println("Some exception occured in method distributeSquare");
    }
    
    // Repaint the top canvas
    topCanvas.repaint();
    //sideCanvas.repaint();
    //frontCanvas.repaint();
  }

  // Method for finding if Cube 2 is contained in Cube 1
  private boolean isCube2InCube1(double side1, double xCent1, double yCent1,
				   double zCent1, double side2, double xCent2,
				   double yCent2, double zCent2) {
    // Find the min and max for each each
    double halfSide = side1/2.0;
    double xMin1 = xCent1 - halfSide;
    double xMax1 = xCent1 + halfSide;
    double yMin1 = yCent1 - halfSide;
    double yMax1 = yCent1 + halfSide;
    double zMin1 = zCent1 - halfSide;
    double zMax1 = zCent1 + halfSide;
    halfSide = side2/2.0;
    double xMin2 = xCent2 - halfSide;
    double xMax2 = xCent2 + halfSide;
    double yMin2 = yCent2 - halfSide;
    double yMax2 = yCent2 + halfSide;
    double zMin2 = zCent2 - halfSide;
    double zMax2 = zCent2 + halfSide;
    if (xMin2 >= xMin1 && xMax2 <= xMax1 && yMin2 >= yMin1 && yMax2 <= yMax1 &&
	zMin2 >= zMin1 && zMax2 <= zMax1) return true;
    return false;
  }

  // Method for finding if two cubes intersect if they are aligned with the
  // co-ordinate axes
  private boolean doCubesIntersect(double side1, double xCent1, double yCent1,
				   double zCent1, double side2, double xCent2,
				   double yCent2, double zCent2) {
    // Find the min and max for each each
    double halfSide = side1/2.0;
    double xMin1 = xCent1 - halfSide;
    double xMax1 = xCent1 + halfSide;
    double yMin1 = yCent1 - halfSide;
    double yMax1 = yCent1 + halfSide;
    double zMin1 = zCent1 - halfSide;
    double zMax1 = zCent1 + halfSide;
    halfSide = side2/2.0;
    double xMin2 = xCent2 - halfSide;
    double xMax2 = xCent2 + halfSide;
    double yMin2 = yCent2 - halfSide;
    double yMax2 = yCent2 + halfSide;
    double zMin2 = zCent2 - halfSide;
    double zMax2 = zCent2 + halfSide;
    double[] xPt = new double[8];
    double[] yPt = new double[8];
    double[] zPt = new double[8];
    xPt[0] = xMin1; yPt[0] = yMin1; zPt[0] = zMin1;
    xPt[1] = xMin1; yPt[1] = yMin1; zPt[1] = zMax1;
    xPt[2] = xMin1; yPt[2] = yMax1; zPt[2] = zMin1;
    xPt[3] = xMin1; yPt[3] = yMax1; zPt[3] = zMax1;
    xPt[4] = xMax1; yPt[4] = yMin1; zPt[4] = zMin1;
    xPt[5] = xMax1; yPt[5] = yMin1; zPt[5] = zMax1;
    xPt[6] = xMax1; yPt[6] = yMax1; zPt[6] = zMin1;
    xPt[7] = xMax1; yPt[7] = yMax1; zPt[7] = zMax1;
    for (int ii = 0; ii < 8; ii++) {
      if (xPt[ii] > xMin2 && xPt[ii] < xMax2 &&
	  yPt[ii] > yMin2 && yPt[ii] < yMax2 &&
	  zPt[ii] > zMin2 && zPt[ii] < zMax2) return true;
    }
    xPt[0] = xMin2; yPt[0] = yMin2; zPt[0] = zMin2;
    xPt[1] = xMin2; yPt[1] = yMin2; zPt[1] = zMax2;
    xPt[2] = xMin2; yPt[2] = yMax2; zPt[2] = zMin2;
    xPt[3] = xMin2; yPt[3] = yMax2; zPt[3] = zMax2;
    xPt[4] = xMax2; yPt[4] = yMin2; zPt[4] = zMin2;
    xPt[5] = xMax2; yPt[5] = yMin2; zPt[5] = zMax2;
    xPt[6] = xMax2; yPt[6] = yMax2; zPt[6] = zMin2;
    xPt[7] = xMax2; yPt[7] = yMax2; zPt[7] = zMax2;
    for (int ii = 0; ii < 8; ii++) {
      if (xPt[ii] > xMin1 && xPt[ii] < xMax1 &&
	  yPt[ii] > yMin1 && yPt[ii] < yMax1 &&
	  zPt[ii] > zMin1 && zPt[ii] < zMax1) return true;
    }
    return false;
  }

  // Method for saving computed particle distribution
  private void saveParticleDist() {

    // Get the name of the file
    File file = null;
    JFileChooser fc = new JFileChooser(new File(".."));
    int returnVal = fc.showSaveDialog(this);
    if (returnVal == JFileChooser.APPROVE_OPTION) file = fc.getSelectedFile();
    if (file == null) return;

    // Create a ParticleList after scaling the co-ordinates and diameters 
    // to 100.0
    double scale = 100.0/d_sideLength;
    ParticleList pl = new ParticleList();
    if (partTypeFlag == CIRCLE) {
      int nofParts = partDiaVector.size();
      for (int ii = 0; ii < nofParts; ii++) {
	Point center = new Point();
	double x = ((Double)partXCentVector.elementAt(ii)).doubleValue();
	double y = ((Double)partYCentVector.elementAt(ii)).doubleValue();
	double radius = ((Double)partDiaVector.elementAt(ii)).doubleValue()/2.0;
	center.setX(x*scale);
	center.setY(y*scale);
	center.setZ(0.0);
	radius *= scale;
	double rot = 0.0;
	pl.addParticle(new Particle(partTypeFlag, radius, rot, center, 0));
      }
    } else if (  partTypeFlag == SPHERE) {
      int nofParts = partDiaVector.size();
      for (int ii = 0; ii < nofParts; ii++) {
	Point center = new Point();
	double x = ((Double)partXCentVector.elementAt(ii)).doubleValue();
	double y = ((Double)partYCentVector.elementAt(ii)).doubleValue();
	double z = ((Double)partZCentVector.elementAt(ii)).doubleValue();
	double radius = ((Double)partDiaVector.elementAt(ii)).doubleValue()/2.0;
	center.setX(x*scale);
	center.setY(y*scale);
	center.setZ(z*scale);
	radius *= scale;
	double rot = 0.0;
	pl.addParticle(new Particle(partTypeFlag, radius, rot, center, 0));
      }
    } else if (partTypeFlag == SQUARE) {
      int nofParts = partSideVector.size();
      for (int ii = 0; ii < nofParts; ii++) {
	Point center = new Point();
	double x = ((Double)partXCentVector.elementAt(ii)).doubleValue();
	double y = ((Double)partYCentVector.elementAt(ii)).doubleValue();
	double radius = 
	  ((Double)partSideVector.elementAt(ii)).doubleValue()/2.0;
	double rot = ((Double)partRotVector.elementAt(ii)).doubleValue()/2.0;
	center.setX(x*scale);
	center.setY(y*scale);
	center.setZ(0.0);
	radius *= scale;
	pl.addParticle(new Particle(partTypeFlag, radius, rot, center, 0));
      }
    } else if (partTypeFlag == CUBE) {
      int nofParts = partSideVector.size();
      for (int ii = 0; ii < nofParts; ii++) {
	Point center = new Point();
	double x = ((Double)partXCentVector.elementAt(ii)).doubleValue();
	double y = ((Double)partYCentVector.elementAt(ii)).doubleValue();
	double z = ((Double)partZCentVector.elementAt(ii)).doubleValue();
	double radius = 
	  ((Double)partSideVector.elementAt(ii)).doubleValue()/2.0;
	double rot = 0.0;
	center.setX(x*scale);
	center.setY(y*scale);
	center.setZ(z*scale);
	radius *= scale;
	pl.addParticle(new Particle(partTypeFlag, radius, rot, center, 0));
      }
    }

    // Save to file
    pl.saveToFile(file, partTypeFlag);

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
                 ylongTick, maxSize;

    // Constructor
    public PlaneCanvas(int width, int height, int type) {
      super(width,height);
      d_type = type;
      maxSize = 10*Math.round((float)(d_sideLength/10.0));
      initialize();
    }

    // initialize
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

    // paint components
    public void paintComponent(Graphics g) {

      // Draw the rules
      drawRule(g);
    }

    // paint the component immediately
    public void paintImmediately() {
      paintImmediately(xmin, xmax, xmax-xmin, ymax-ymin);
    }
    public void paintImmediately(int x, int y, int w, int h) {
      Graphics g = getGraphics();
      super.paintImmediately(x, y, w, h);
      drawRule(g);
    }

    // Method to draw the Rule
    private void drawRule(Graphics g) {

      // Get the particle size data to fix the dimensions of the axes
      maxSize = 10*Math.round((float)Math.ceil((d_sideLength/10.0)));
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

    // Get the screen co=ordinates of a world point
    protected int getXScreenCoord(double coord) {
      return xmin+(int) (coord/(double)maxSize*(double)(xmax-xmin));
    }
    protected int getYScreenCoord(double coord) {
      return ymax-(int) (coord/(double)maxSize*(double)(ymax-ymin));
    }
    protected int getXScreenLength(double length) {
      return (int) (length/(double) maxSize *(double)(xmax-xmin));
    }
    protected int getYScreenLength(double length) {
      return (int) (length/(double) maxSize *(double)(ymax-ymin));
    }
  }

  //**************************************************************************
  // Class   : TopCanvas
  // Purpose : Draws a LightWeightCanvas for displaying the planes.
  //           More detailed canvases ar derived from this.
  //**************************************************************************
  private final class TopCanvas extends PlaneCanvas {

    // Data

    // Constructor
    public TopCanvas(int width, int height) {
      super(width,height,TOP);
      initialize();
    }

    // initialize
    private void initialize() {
    }

    // paint components
    public void paintComponent(Graphics g) {
      super.paintComponent(g);
      if (partTypeFlag == CIRCLE) 
	drawCircles(g);
      else if (partTypeFlag == SQUARE) {
	//drawFilledSubcells(g);
	drawBoxes(g);
	//drawSquares(g);
      }
      else if (partTypeFlag == SPHERE)
	drawSpheres(g);
      else if (partTypeFlag == CUBE)
	drawCubes(g);
    }

    // paint the component immediately
    public void paintImmediately() {
      paintImmediately(xmin, xmax, xmax-xmin, ymax-ymin);
    }
    public void paintImmediately(int x, int y, int w, int h) {
      Graphics g = getGraphics();
      super.paintImmediately(x, y, w, h);
      if (partTypeFlag == CIRCLE) 
	drawCircles(g);
      else if (partTypeFlag == SQUARE) 
	drawSquares(g);
      else if (partTypeFlag == SPHERE)
	drawSpheres(g);
      else if (partTypeFlag == CUBE)
	drawCubes(g);
    }
    

    // Draw the circles
    private void drawCircles(Graphics g) {
      if (partDiaVector == null) return;
      int size = partDiaVector.size();
      double dia, xCent, yCent;
      for (int ii = 0; ii < size; ii++) {
	dia = ((Double)(partDiaVector.elementAt(ii))).doubleValue();
	xCent = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	int radXScreen = getXScreenLength(dia/2.0);
	int radYScreen = getYScreenLength(dia/2.0);
	int xCentScreen = getXScreenCoord(xCent);
	int yCentScreen = getYScreenCoord(yCent);
	g.setColor(new Color(184,119,27));
	g.fillOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
	g.setColor(new Color(0,0,0));
	g.drawOval(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
      }
    }

    // Draw the squares
    private void drawSquares(Graphics g) {
      try {
      if (partSideVector == null) return;
      int size = partSideVector.size();
      int[] xCoord = new int[4];
      int[] yCoord = new int[4];
      for (int ii = 0; ii < size; ii++) {
	xCoord[0] = 
	  getXScreenCoord(((Point) partPt1Vector.elementAt(ii)).getX());
	xCoord[1] = 
	  getXScreenCoord(((Point) partPt2Vector.elementAt(ii)).getX());
	xCoord[2] = 
	  getXScreenCoord(((Point) partPt3Vector.elementAt(ii)).getX());
	xCoord[3] = 
	  getXScreenCoord(((Point) partPt4Vector.elementAt(ii)).getX());
	yCoord[0] = 
	  getYScreenCoord(((Point) partPt1Vector.elementAt(ii)).getY());
	yCoord[1] = 
	  getYScreenCoord(((Point) partPt2Vector.elementAt(ii)).getY());
	yCoord[2] = 
	  getYScreenCoord(((Point) partPt3Vector.elementAt(ii)).getY());
	yCoord[3] = 
	  getYScreenCoord(((Point) partPt4Vector.elementAt(ii)).getY());
	g.setColor(new Color(184,119,27));
	g.fillPolygon(xCoord, yCoord, 4);
	g.setColor(new Color(0,0,0));
	g.drawPolygon(xCoord, yCoord, 4);
      }
      } catch (Exception e) {
	System.out.println("Exception painting squares in TopCanvas");
      }
    }

    // Draw the spheres
    private void drawSpheres(Graphics g) {
      if (partDiaVector == null) return;

      // Store the particle data
      int size = partDiaVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partDiaVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	zCent[ii] = ((Double)(partZCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of ascending Z Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyZCent = zCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && zCent[ii] > keyZCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  zCent[ii+1] = zCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	zCent[ii+1] = keyZCent;
	dia[ii+1] = keyDia;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
	int radXScreen = getXScreenLength(dia[ii]/2.0);
	int radYScreen = getYScreenLength(dia[ii]/2.0);
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

    // Draw the cubes
    private void drawCubes(Graphics g) {
      if (partSideVector == null) return;

      // Store the particle data
      int size = partSideVector.size();
      double[] side = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	side[ii] = ((Double)(partSideVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	zCent[ii] = ((Double)(partZCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of ascending Z Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyZCent = zCent[jj];
	double keySide = side[jj];
	int ii = jj-1;
	while (ii >= 0 && zCent[ii] > keyZCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  zCent[ii+1] = zCent[ii];
	  side[ii+1] = side[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	zCent[ii+1] = keyZCent;
	side[ii+1] = keySide;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
	int radXScreen = getXScreenLength(side[ii]/2.0);
	int radYScreen = getYScreenLength(side[ii]/2.0);
	int xCentScreen = getXScreenCoord(xCent[ii]);
	int yCentScreen = getYScreenCoord(yCent[ii]);
	g.setColor(new Color(184,119,27));
	g.fillRect(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
	g.setColor(new Color(0,0,0));
	g.drawRect(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
      }
    }
    // Method for drawing boxes
    public void drawBoxes(Graphics g) {
      int nofBoxes = 0;
      if ((nofBoxes = d_boxList.size()) <= 0) return;
      for (int ii = 0; ii < nofBoxes; ii++) {
	Box bx = (Box) d_boxList.get(ii);
	Point lower = bx.getLower();
	Point upper = bx.getUpper();
	g.setColor(new Color(184,119,54));
	double xmn = lower.getX();
	double xmx = upper.getX();
	double ymn = lower.getY();
	double ymx = upper.getY();
	int x1 = getXScreenCoord(xmn);
	int y2 = getYScreenCoord(ymn);
	int x2 = getXScreenCoord(xmx);
	int y1 = getYScreenCoord(ymx);
	//g.fillRect(x1, y1, x2-x1, y2-y1);
	g.setColor(new Color(0,0,0));
	g.drawRect(x1, y1, x2-x1, y2-y1);
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

    // Constructor
    public SideCanvas(int width, int height) {
      super(width,height,SIDE);
      initialize();
    }

    // initialize
    private void initialize() {
    }

    // paint components
    public void paintComponent(Graphics g) {
      super.paintComponent(g);
      if (partTypeFlag == SPHERE) 
	drawSpheres(g);
      else if (partTypeFlag == CUBE) 
	drawCubes(g);
    }

    // Draw the spheres
    private void drawSpheres(Graphics g) {
      if (partDiaVector == null) return;

      // Store the particle data
      int size = partDiaVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partDiaVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	zCent[ii] = ((Double)(partZCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyZCent = zCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && xCent[ii] > keyXCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  zCent[ii+1] = zCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	zCent[ii+1] = keyZCent;
	dia[ii+1] = keyDia;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
	int radXScreen = getXScreenLength(dia[ii]/2.0);
	int radYScreen = getYScreenLength(dia[ii]/2.0);
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

    // Draw the cubes
    private void drawCubes(Graphics g) {
      if (partSideVector == null) return;

      // Store the particle data
      int size = partSideVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partSideVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	zCent[ii] = ((Double)(partZCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyZCent = zCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && xCent[ii] > keyXCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  zCent[ii+1] = zCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	zCent[ii+1] = keyZCent;
	dia[ii+1] = keyDia;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
	int radXScreen = getXScreenLength(dia[ii]/2.0);
	int radYScreen = getYScreenLength(dia[ii]/2.0);
	int xCentScreen = getXScreenCoord(yCent[ii]);
	int yCentScreen = getYScreenCoord(zCent[ii]);
	g.setColor(new Color(184,119,27));
	g.fillRect(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
	g.setColor(new Color(0,0,0));
	g.drawRect(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
      }
    }

    // Draw the cylinders corresponding to the circles with min X
    // cylinders first
    private void drawCylinder(Graphics g) {
      if (partDiaVector == null) return;

      // Store the particle data
      int size = partDiaVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partDiaVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && xCent[ii] > keyXCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	dia[ii+1] = keyDia;
      }

      // Draw the lines next
      int blue = 216;
      for (int ii = 0; ii < size; ii++) {
	int radScreen = getXScreenLength(dia[ii]/2.0);
	int centScreen = getXScreenCoord(yCent[ii]);
	int quo = ii/100;
	if (quo >= 7) blue = 27;
	else if (quo == 6) blue = 54;
	else if (quo == 5) blue = 81;
	else if (quo == 4) blue = 108;
	else if (quo == 3) blue = 135;
	else if (quo == 2) blue = 162;
	else if (quo == 1) blue = 189;
	g.setColor(new Color(184,119,blue));
	g.fillRect(centScreen-radScreen,ymin,2*radScreen,ymax-ymin);
	g.setColor(new Color(0,0,0));
	g.drawRect(centScreen-radScreen,ymin,2*radScreen,ymax-ymin);
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

    // Constructor
    public FrontCanvas(int width, int height) {
      super(width,height,GenerateParticlePanel.FRONT);
      initialize();
    }

    // initialize
    private void initialize() {
    }

    // paint components
    public void paintComponent(Graphics g) {
      super.paintComponent(g);
      if (partTypeFlag == SPHERE) 
	drawSpheres(g);
      else if (partTypeFlag == CUBE) 
	drawCubes(g);
      if (partTypeFlag == SQUARE) {
	//drawFilledSubcells(g);
	drawBoxes(g);
      }
    }

    // Draw the spheres
    private void drawSpheres(Graphics g) {
      if (partDiaVector == null) return;

      // Store the particle data
      int size = partDiaVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partDiaVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	zCent[ii] = ((Double)(partZCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of descending Y Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyZCent = zCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && yCent[ii] < keyYCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  zCent[ii+1] = zCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	zCent[ii+1] = keyZCent;
	dia[ii+1] = keyDia;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
	int radXScreen = getXScreenLength(dia[ii]/2.0);
	int radYScreen = getYScreenLength(dia[ii]/2.0);
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

    // Draw the cubes
    private void drawCubes(Graphics g) {
      if (partDiaVector == null) return;

      // Store the particle data
      int size = partSideVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      double[] zCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partSideVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
	zCent[ii] = ((Double)(partZCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of descending Y Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyZCent = zCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && yCent[ii] < keyYCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  zCent[ii+1] = zCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	zCent[ii+1] = keyZCent;
	dia[ii+1] = keyDia;
      }

      // Draw the circles next
      for (int ii = 0; ii < size; ii++) {
	int radXScreen = getXScreenLength(dia[ii]/2.0);
	int radYScreen = getYScreenLength(dia[ii]/2.0);
	int xCentScreen = getXScreenCoord(xCent[ii]);
	int yCentScreen = getYScreenCoord(zCent[ii]);
	g.setColor(new Color(184,119,27));
	g.fillRect(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
	g.setColor(new Color(0,0,0));
	g.drawRect(xCentScreen-radXScreen,yCentScreen-radYScreen,
		   2*radXScreen, 2*radYScreen);
      }
    }

    // Draw the cylinders corresponding to the circles with max Y
    // cylinders first
    private void drawCylinder(Graphics g) {
      if (partDiaVector == null) return;

      // Store the particle data
      int size = partDiaVector.size();
      double[] dia = new double[size]; 
      double[] xCent = new double[size];
      double[] yCent = new double[size];
      for (int ii = 0; ii < size; ii++) {
	dia[ii] = ((Double)(partDiaVector.elementAt(ii))).doubleValue();
	xCent[ii] = ((Double)(partXCentVector.elementAt(ii))).doubleValue();
	yCent[ii] = ((Double)(partYCentVector.elementAt(ii))).doubleValue();
      }

      // sort the particle data in order of ascending X Coord
      for (int jj = 1; jj < size; jj++) {
	double keyXCent = xCent[jj];
	double keyYCent = yCent[jj];
	double keyDia = dia[jj];
	int ii = jj-1;
	while (ii >= 0 && yCent[ii] < keyYCent) {
	  xCent[ii+1] = xCent[ii];
	  yCent[ii+1] = yCent[ii];
	  dia[ii+1] = dia[ii];
	  ii--;
	}
	xCent[ii+1] = keyXCent;
	yCent[ii+1] = keyYCent;
	dia[ii+1] = keyDia;
      }

      // Draw the lines next
      int blue = 216;
      for (int ii = 0; ii < size; ii++) {
	int radScreen = getXScreenLength(dia[ii]/2.0);
	int centScreen = getXScreenCoord(xCent[ii]);
	int quo = ii/100;
	if (quo >= 7) blue = 27;
	else if (quo == 6) blue = 54;
	else if (quo == 5) blue = 81;
	else if (quo == 4) blue = 108;
	else if (quo == 3) blue = 135;
	else if (quo == 2) blue = 162;
	else if (quo == 1) blue = 189;
	g.setColor(new Color(184,119,blue));
	g.fillRect(centScreen-radScreen,ymin,2*radScreen,ymax-ymin);
	g.setColor(new Color(0,0,0));
	g.drawRect(centScreen-radScreen,ymin,2*radScreen,ymax-ymin);
      }
    }

    // Method for filling subcells
    /*
    public void drawFilledSubcells(Graphics g) {
      int nofSubcells = d_subcellList.getSize();
      System.out.println("In drawFilledSubcells # : "+nofSubcells);
      if (nofSubcells <= 0) return;
      for (int ii = 0; ii < nofSubcells; ii++) {
	int matCode = d_subcellList.getSubcellMat(ii);
	if (matCode == 1) {
	  g.setColor(new Color(0,255,0));
	  Vector coords = d_subcellList.getSubcellCoords(ii);
	  double xmn = ((Double) coords.elementAt(0)).doubleValue();
	  double xmx = ((Double) coords.elementAt(1)).doubleValue();
	  double ymn = ((Double) coords.elementAt(2)).doubleValue();
	  double ymx = ((Double) coords.elementAt(3)).doubleValue();
	  int x1 = getXScreenCoord(xmn);
	  int y2 = getYScreenCoord(ymn);
	  int x2 = getXScreenCoord(xmx);
	  int y1 = getYScreenCoord(ymx);
	  g.fillRect(x1, y1, x2-x1, y2-y1);
	}
      }
    }
    
    */
    // Method for drawing boxes
    public void drawBoxes(Graphics g) {
      int nofBoxes = 0;
      if ((nofBoxes = d_boxList.size()) <= 0) return;
      for (int ii = 0; ii < nofBoxes; ii++) {
	Box bx = (Box) d_boxList.get(ii);
	Point lower = bx.getLower();
	Point upper = bx.getUpper();
	g.setColor(new Color(184,119,54));
	double xmn = lower.getX();
	double xmx = upper.getX();
	double ymn = lower.getY();
	double ymx = upper.getY();
	int x1 = getXScreenCoord(xmn);
	int y2 = getYScreenCoord(ymn);
	int x2 = getXScreenCoord(xmx);
	int y1 = getYScreenCoord(ymx);
	//g.fillRect(x1, y1, x2-x1, y2-y1);
	g.setColor(new Color(0,0,0));
	g.drawRect(x1, y1, x2-x1, y2-y1);
      }
    }
  }

  //**************************************************************************
  // Class   : OrthoCanvas
  // Purpose : Draws a LightWeightCanvas for displaying the planes.
  //           More detailed canvases ar derived from this.
  //**************************************************************************
  private class OrthoCanvas extends LightWeightCanvas {

    // Data

    // Constructor
    public OrthoCanvas(int width, int height) {
      super(width,height);
      initialize();
    }

    // initialize
    private void initialize() {
    }

    // paint components
    public void paintComponent(Graphics g) {
    }
  }

  //**************************************************************************
  // Class   : Box
  // Purpose : Stores the lower left and upper right corners of a box
  //**************************************************************************
  private class Box extends Object {

    // Data
    private Point box_lowerLeft;
    private Point box_upperRight;

    // Constructor
    public Box(Point lower, Point upper) {
      box_lowerLeft = lower;
      box_upperRight = upper;
    }

    // Get
    Point getLower() { return box_lowerLeft; }
    Point getUpper() { return box_upperRight; }
  }
}
