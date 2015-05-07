//**************************************************************************
// Program : Packing.java
// Purpose : Given a discrete particle size distribution this computes
//           a tight packing of the particles and imposes periodic 
//           boundary conditions.
//           The algorithm is :
//             1) Distribute the particles in a large cube divided into
//                cells such that the cell size is bigger than the 
//                largest particle.
//             2) Scale the position of the particles based on the
//                smallest distance between two particles.
//             3) Then give the particles a random velocity and direction
//                and allow them to settle.
//             4) Any particles that move out of the box come back on the
//                other side.
// Original Author in FORTRAN : Oleksiy Byutner
// Author in Java : Biswajit Banerjee
// Date    : 07/13/2000
// Mods    :
//**************************************************************************
// $Id: $

//************ IMPORTS **************
import java.util.Random;
import java.util.Vector;
import java.io.*;

//**************************************************************************
// Class   : Packing
//**************************************************************************
public class Packing {

  /**
   *  Const data
   */
  public static final int TWO_DIM = 0;
  public static final int THREE_DIM = 1;
  /**
   *  Private data
   */
  private int d_dim;
  private ParticleSize d_partSizeDist;
  private int d_nofSizeFrac;
  private int d_nofParticles;
  private double d_maxRad;
  private double d_totalVolume;
  private PParticle[] d_particle;
  private double d_boxSize;

  /**
   *  Constructor
   */
  public Packing(int dim, ParticleSize partSizeDist) {
    d_dim = dim;
    d_partSizeDist = partSizeDist;
    d_nofSizeFrac = partSizeDist.nofSizesCalc;
    d_nofParticles = 0;
    d_maxRad = -1.0;
    d_totalVolume = 0.0;
    d_particle = null;
    d_boxSize = 0.0;
  }

  /**
   *  Get the results
   */
  public int nofParticles() { return d_nofParticles; }
  public double radius(int index) {
    if (index > -1 && index < d_nofParticles)
      return d_particle[index].radius();
    return -1.0;
  }
  public double xLoc(int index) {
    if (index > -1 && index < d_nofParticles)
      return d_particle[index].center().x();
    return -1.0;
  }
  public double yLoc(int index) {
    if (index > -1 && index < d_nofParticles)
      return d_particle[index].center().y();
    return -1.0;
  }
  public double zLoc(int index) {
    if (index > -1 && index < d_nofParticles)
      return d_particle[index].center().z();
    return -1.0;
  }
  public double boxSize() { return d_boxSize; }

  /**
   *  Create a packing
   */
  public void createPacking(int maxSteps, int maxTry, double maxVolFrac) {
    setInitialData();
    placeInInitialGrid();
    rattleBox(maxSteps, maxTry, maxVolFrac);
  }

  /**
   *  Find the maximum diameter of particles, the number of particles in 2D
   *  and 3D, and the total volme in 2d and 3D
   */
  private void setInitialData() {
    for (int ii = 0; ii < d_nofSizeFrac; ii++) {
      double dia = d_partSizeDist.sizeCalc[ii];
      if (dia > d_maxRad) d_maxRad = dia;
      int nofParts = 0;
      if (d_dim == TWO_DIM) {
        nofParts = d_partSizeDist.freq2DCalc[ii];
        d_totalVolume += (double)nofParts*Math.pow(dia,2);
      } else {
        nofParts = d_partSizeDist.freq3DCalc[ii];
        d_totalVolume += (double)nofParts*Math.pow(dia,3);
      }
      d_nofParticles += nofParts;
    }
    d_maxRad /= 2.0;
    if (d_dim == TWO_DIM) 
      d_totalVolume *= (Math.PI/4);
    else
      d_totalVolume *= (Math.PI/6);
    d_particle = new PParticle[d_nofParticles];
    for (int ii = 0; ii < d_nofParticles; ii++) d_particle[ii] = new PParticle();

    // Debug
    System.out.println("MaxRad = "+d_maxRad+" totalVol = "+d_totalVolume+
		       " No of Particles = "+d_nofParticles);
  }

  /**
   *  Create a grid to put the particles and place the particles
   *  one by one into random locations in the grid 
   */
  private void placeInInitialGrid() {

    // Create the initial grid
    int nofGridCells = 0;
    if (d_dim == TWO_DIM) 
      nofGridCells = (int) Math.ceil(Math.pow(d_nofParticles, (1.0/2.0))) + 1;
    else 
      nofGridCells = (int) Math.ceil(Math.pow(d_nofParticles, (1.0/3.0))) + 1;
    double maxRad = 1.05*d_maxRad;
    double maxDia = 2.0*maxRad;
    d_boxSize = maxDia*(double) nofGridCells;
    double[] xx = new double[nofGridCells];
    double[] yy = new double[nofGridCells];
    double[] zz = new double[nofGridCells];

    // Set the particle inserted in grid location flag to false
    boolean[][][] inserted ;
    if (d_dim == TWO_DIM) 
      inserted = new boolean[nofGridCells][nofGridCells][1];
    else
      inserted = new boolean[nofGridCells][nofGridCells][nofGridCells];
    for (int ii = 0; ii < nofGridCells; ii++) {
      for (int jj = 0; jj < nofGridCells; jj++) {
	if (d_dim == TWO_DIM) 
	  inserted[ii][jj][0] = false;
	else {
	  for (int kk = 0; kk < nofGridCells; kk++) {
	    inserted[ii][jj][kk] = false;
	  }
	}
      }
      double xi = maxRad + ii*maxDia;
      xx[ii] = xi;
      yy[ii] = xi;
      if (d_dim == TWO_DIM)
	zz[ii] = 0.0;
      else
	zz[ii] = xi;
    }

    // Place the particles one by one into the grid
    int particleNo = 0;
    for (int ii = 0; ii < d_nofSizeFrac; ii++) {
      double dia = d_partSizeDist.sizeCalc[ii];
      double radius = dia/2.0;
      int number;
      if (d_dim == TWO_DIM)
	number = d_partSizeDist.freq2DCalc[ii];
      else
	number = d_partSizeDist.freq3DCalc[ii];
      for (int jj = 0; jj < number; jj++) {
	// Generate two random indices between 0 and nofGridCells-1
	int xIndex, yIndex, zIndex;
	Random rand = new Random();
	do {
	  xIndex = rand.nextInt(nofGridCells);
	  yIndex = rand.nextInt(nofGridCells);
	  if (d_dim == TWO_DIM)
	    zIndex = 0;
	  else
	    zIndex = rand.nextInt(nofGridCells);
	} while(inserted[xIndex][yIndex][zIndex]);
	// Save the location and radius of the particle
	d_particle[particleNo].radius(radius);
	if (d_dim == TWO_DIM)
	  d_particle[particleNo].center(xx[xIndex], yy[yIndex], 0);
	else
	  d_particle[particleNo].center(xx[xIndex], yy[yIndex], zz[zIndex]);

	inserted[xIndex][yIndex][zIndex] = true;
	++particleNo;
      }
    }

    // Debug
    if (d_dim == TWO_DIM) {
      double volFrac = d_totalVolume/Math.pow(d_boxSize, 2.0);
      System.out.println("Vol Frac = "+volFrac+" Box Size = "+d_boxSize);
    } else {
      double volFrac = d_totalVolume/Math.pow(d_boxSize, 3.0);
      System.out.println("Vol Frac = "+volFrac+" Box Size = "+d_boxSize);
    }
    String fileName = new String("./ballLoc0");
    File outFile = new File(fileName);
    try {
      FileWriter fw = new FileWriter(outFile);
      PrintWriter pw = new PrintWriter(fw);
      for (int ii = 0; ii < d_nofParticles; ii++) {
	double rad = d_particle[ii].radius();
	double x = d_particle[ii].center().x();
	double y = d_particle[ii].center().y();
	double z = d_particle[ii].center().z();
	pw.println(rad+" "+x+" "+y+" "+z);
      }
      pw.close();
      fw.close();
    } catch (Exception e) {
      System.out.println("Could not write to "+fileName);
    }
  }

  /**
   *  Rattle the box and the particles
   */
  public void rattleBox(int maxSteps, int maxTry, double maxVolFrac) {

    // Setup the simulation
    int stepCount = 1; 
    double volFrac = 0.0;
    double stepSize = 10.0;


    maxSteps = 2;
    do {

      System.out.println("Step no = "+stepCount);

      // Create the neighbor list
      createNeighborList();

      // Calculate the scale factor for scaling the co-ordinates
      double scale = calculateScaleFactor();
      System.out.println("ScaleFactor = "+scale);

      while (stepSize > 1.0e-4) {

	// Rescale all radii
	rescaleRadii(scale);

	// Create the neighbor list again
	createNeighborList();
      
	// Move neighbors and particles
	stepSize = moveParticles(maxTry);

	// If the step size is too small .. scale up the radii and create
	// a new neighbor list
	if (stepSize <= 1.0e-4) scale *= 1.1;
      }

      if (d_dim == TWO_DIM) 
	volFrac = d_totalVolume/Math.pow(d_boxSize, 2.0);
      else
	volFrac = d_totalVolume/Math.pow(d_boxSize, 3.0);

      if (stepCount%500 == 0) {
	System.out.println("Steps = "+stepCount+" Vol frac = "+volFrac+
			   " Box Size = "+d_boxSize);
	// Debug
	String fileName = new String("./ballLoc"+stepCount);
	File outFile = new File(fileName);
	try {
	  FileWriter fw = new FileWriter(outFile);
	  PrintWriter pw = new PrintWriter(fw);
	  for (int ii = 0; ii < d_nofParticles; ii++) {
	    double rad = d_particle[ii].radius();
	    double x = d_particle[ii].center().x();
	    double y = d_particle[ii].center().y();
	    double z = d_particle[ii].center().z();
	    pw.println(rad+" "+x+" "+y+" "+z);
	  }
	  pw.close();
	  fw.close();
	} catch (Exception e) {
	  System.out.println("Could not write to "+fileName);
	}
      }

      // Debug statements
      if (stepCount%1000 == 0)
	System.out.println("Steps = "+stepCount+" Vol frac = "+volFrac);

      // Increment step count
      ++stepCount;

    } while (stepCount < maxSteps && volFrac < maxVolFrac);
  }

  //
  // Create a list of neighbors
  //
  private void createNeighborList() {
    
    // Clear the previous list
    for (int ii = 0; ii < d_nofParticles; ii++) {
      d_particle[ii].clearNeighborList();
    }

    // Look at a distance equal to the max radius
    for (int ii = 0; ii < d_nofParticles-1; ii++) {
      PParticle p1 = d_particle[ii];
      for (int jj = ii+1; jj < d_nofParticles; jj++) {
	PParticle p2 = d_particle[jj];
	double dist = p1.distance(p2);
	if (Math.abs(dist) < 2.0*d_maxRad) {
	  d_particle[ii].addNeighbor(p2);
	  d_particle[jj].addNeighbor(p1);
	}
      }
    }

    // Find the smallest distance and nearest neighbor
    for (int ii = 0; ii < d_nofParticles; ii++) {
      int numNei = d_particle[ii].nofNeighbors();
      PParticle p1 = d_particle[ii];
      PParticle nearest = p1;
      double smallDist  = 1.0e10;
      for (int jj = 0; jj < numNei; jj++) {
	PParticle p2 = p1.neighbor(jj);
	double dist = p1.distance(p2);
	if (dist < smallDist) {
	  smallDist = dist;
	  nearest = p2;
	}
      }
      d_particle[ii].nearestParticle(nearest, smallDist);
    }

    
    for (int ii = 0; ii < d_nofParticles; ii++) {
      System.out.print("Particle = "+ii);
      //int numNei = d_particle[ii].nofNeighbors();
      //for (int jj = 0; jj < numNei; jj++) 
	//System.out.print(" "+ d_particle[ii].neighbor(jj)+" ");
      System.out.println(" ");
      System.out.println("Nearest Neighbor = "+d_particle[ii].nearestParticle());
      System.out.println("Smallest Dist = "+d_particle[ii].smallestDistance());
    }
    
  }

  //
  // Calculate the scale factor for scaling radius
  //
  public double calculateScaleFactor() {
    double minScale = 1.0e10;
    double scale = 0.0;

    for (int ii = 0; ii < d_nofParticles-1; ii++) {
      int nofNeighbors = d_particle[ii].nofNeighbors();
      double iRadius = d_particle[ii].radius();
      if (nofNeighbors > 0) {
	double smallDist = d_particle[ii].smallestDistance();
	PParticle nearest = d_particle[ii].nearestParticle();
	double jRadius = nearest.radius();
	double radSum = iRadius + jRadius;
	scale = (smallDist+radSum)/radSum;
	if (scale < minScale) minScale = scale;
	if (scale < 1.0) {
	  System.out.println("In SCALE: Distance "+smallDist+" between "+ii+
			     " and "+nearest+" is less than the sum"+
			     " of their radii "+iRadius+"+"+jRadius);
	}
      }
    }
    return 1.0+(minScale-1.0)*0.95;
  }

  //
  //  Rescale the radii 
  //
  private void rescaleRadii(double scale) {
    for (int ii = 0; ii < d_nofParticles; ii++) {
      d_particle[ii].scaleRadius(scale);
    }
  }

  //
  //  Apply an acceptable random movement to each particle
  //
  private double moveParticles(int maxTry) {

    // Create a random number generator and set the step size
    Random rand = new Random();
    double stepSize = 0.0;

    for (int ii = 0; ii < d_nofParticles; ii++) {
      
      System.out.println("Tring to move Particle "+ii);
      // Get the number of neighbors to be tested
      // Also particle co-ordinates and radius
      PParticle p1 = d_particle[ii];
      stepSize = 0.1*p1.radius();
      int nofNeighbors = p1.nofNeighbors();
      double xCent = p1.center().x();
      double yCent = p1.center().y();
      double zCent = p1.center().z();
      double rad1 = p1.radius();

      double x1 = xCent;
      double y1 = yCent;
      double z1 = zCent;

      boolean tryAgain = false;
      boolean moveParticle = true;
      int nofTry = 0;

      do {

	// Set tryagain
	tryAgain = false;

	// If the number of tries is greter than maxtry reduce the step size
	// reset and try again
	if (nofTry > maxTry)  { 
	  stepSize /= 1.1; 
	  if (stepSize < 1.0e-4) {
	    moveParticle = false;
	    break;
	  }
	  nofTry = 0; 
	}

	// Get a random movement (-stepSize to stepSize)
	double moveX = stepSize*(2.0*rand.nextDouble()-1.0);
	double moveY = stepSize*(2.0*rand.nextDouble()-1.0);
	double moveZ = 0.0;
	if (d_dim == THREE_DIM) moveZ = stepSize*(2.0*rand.nextDouble()-1.0);

	// Add the random movement to the co-ordinates of the center
	x1 = xCent+moveX; 
	y1 = yCent+moveY; 
	z1 = zCent+moveZ;

	// Create a PPoint for the new center
	PPoint cent1 = new PPoint(x1, y1, z1);

	// Check if this movement leads to intersection with neighbors
	for (int jj = 0; jj < nofNeighbors; jj++) {
	  PParticle p2 = p1.neighbor(jj);
	  double rad2 = p2.radius();
	  PPoint cent2 = p2.center();
	  double dist = cent1.distance(cent2);
	  double sumRad = rad1 + rad2;
	  if (dist < sumRad) {
	    ++nofTry;
	    tryAgain = true;
	    break;
	  }
	  //System.out.println("Try# = "+nofTry+"moves="+moveX+" "+moveY+
	  //	       " "+moveZ+" StepSize = "+stepSize+" Distance = "+
	  //	       dist+ " Rad sum = "+sumRad);
	}
      } while (tryAgain);

      if (moveParticle) {
	// Update the co-odinates of the particle
	if (x1 > d_boxSize) x1 -= d_boxSize;
	if (y1 > d_boxSize) y1 -= d_boxSize;
	if (x1 < 0.0) x1 += d_boxSize;
	if (y1 < 0.0) y1 += d_boxSize;
	if (d_dim == THREE_DIM) {
	  if (z1 < 0.0) z1 += d_boxSize;
	  if (z1 > d_boxSize) z1 -= d_boxSize;
	}
	d_particle[ii].center(x1, y1, z1);
	System.out.println("Particle "+ii+" moved");
      } else {
	System.out.println("Particle "+ii+" not moved");
      }
    }
    return stepSize;
  }

  //***********************************************************************
  // Private class Particle
  //***********************************************************************
  private class PParticle {
    
    // Private data
    private double p_radius;
    private PPoint  p_cent;
    private Vector p_neighborList;
    private double p_smallestDist;
    private PParticle p_nearestParticle;

    // Constructor
    PParticle() {
      p_radius = 0.0;
      p_cent = null;
      p_neighborList = new Vector();
      p_smallestDist = 0.0;
      p_nearestParticle = null;
    }
    PParticle(double rad, PPoint cent) {
      p_radius = rad;
      p_cent = cent;
      p_neighborList = new Vector();
      p_smallestDist = 0.0;
      p_nearestParticle = null;
    }

    // Get
    double radius() { return p_radius; }
    PPoint center() { return p_cent; }
    PParticle neighbor(int jj) {
      return (PParticle) p_neighborList.get(jj);
    }
    int nofNeighbors() { return p_neighborList.size(); }
    PParticle nearestParticle() { return p_nearestParticle; }
    double smallestDistance() { return p_smallestDist; }
    
    // Set
    void radius(double rad) { p_radius = rad; }
    void center(double x, double y, double z) { 
      p_cent = new PPoint(x, y, z); 
    }
    void nearestParticle(PParticle p, double d) { 
      p_nearestParticle = p; 
      p_smallestDist = d; 
    }

    // Scale
    void scaleRadius(double factor) { p_radius *= factor; }
    void scaleCenter(double factor) { p_cent.scale(factor); }

    // Add a near Particle
    void addNeighbor(PParticle particle) {
      p_neighborList.add(particle);
    }

    // Delete the list of near particles
    void clearNeighborList() { 
      if (p_neighborList.size() > 0) p_neighborList.clear(); 
    }

    // Find the distance between two particles
    // assuming periodic bcs
    double distance(PParticle particle) {
      double rad1 = p_radius;
      double rad2 = particle.p_radius;
      double centDist = p_cent.distance(particle.p_cent);
      double dist = centDist - rad1 - rad2;
      return dist;
    }
  }
  
  //***********************************************************************
  // Private class Point
  //***********************************************************************
  private class PPoint {
    
    // Private data
    private double p_x;
    private double p_y;
    private double p_z;

    // Constructor
    PPoint(double x, double y, double z) {
      p_x = x; p_y = y; p_z = z;
    }

    // Get
    double x() { return p_x; }
    double y() { return p_y; }
    double z() { return p_z; }

    // Scale
    void scale(double factor) {
      p_x *= factor;
      p_y *= factor;
      p_z *= factor;
    }

    // Distance (assuming periodic bcs)
    double distance(PPoint pt) {
      double dx = p_x - pt.p_x;
      double dy = p_y - pt.p_y;
      double dz = p_z - pt.p_z;
      double halfBox = d_boxSize/2.0;
      if (Math.abs(dx) > halfBox) dx -= sign(d_boxSize, dx);
      if (Math.abs(dy) > halfBox) dy -= sign(d_boxSize, dy);
      if (Math.abs(dz) > halfBox) dz -= sign(d_boxSize, dz);
      double dd = Math.sqrt(dx*dx+dy*dy+dz*dz);
      return dd;
    }
  }
  
  /**
   *  dsign
   */
  private double sign(double a, double b) {
    if ((a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0)) return -a;
    return a;
  }

} // End class Packing

//
// $ Log: $
//
