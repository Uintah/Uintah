//**************************************************************************
// Class   : CreateGeomPiecePanel
// Purpose : Panel to create, delete, and modify geometry pieces
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import javax.swing.*;

public class CreateGeomPiecePanel extends JPanel 
                                  implements ActionListener {

  // Data
  private InputGeometryPanel d_parent = null;
  private Vector d_geomPiece = null;
  private ParticleList d_partList = null;
  private boolean d_partGeomPieceExists = false;

  // Components
  private JButton addButton = null;
  private JButton delButton = null;
  private JTabbedPane geomPieceTabPane = null;

  //-------------------------------------------------------------------------
  // Constructor
  //-------------------------------------------------------------------------
  public CreateGeomPiecePanel(boolean usePartList,
                              ParticleList partList,
                              Vector geomPiece,
                              InputGeometryPanel parent) {

    // Initialize
    d_geomPiece = geomPiece;
    d_partList = partList;
    d_partGeomPieceExists = false;

    // Save the arguments
    d_parent = parent;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create a panel for the buttons and the buttons
    JPanel panel = new JPanel(new GridLayout(1,0));

    addButton = new JButton("Create Geom Piece");
    addButton.setActionCommand("add");
    addButton.addActionListener(this);
    panel.add(addButton);
      
    delButton = new JButton("Delete Geom Piece");
    delButton.setActionCommand("del");
    delButton.addActionListener(this);
    panel.add(delButton);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(panel, gbc);
    add(panel);

    // Create a tabbed pane for the geometrypieces
    geomPieceTabPane = new JTabbedPane();

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(geomPieceTabPane, gbc);
    add(geomPieceTabPane);
  }


  //-------------------------------------------------------------------------
  // Actions performed when a button is pressed
  //-------------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    if (e.getActionCommand() == "add") {
      String tabName = new String("Object ");
      GeomPiecePanel geomPiecePanel = new GeomPiecePanel(this);
      geomPieceTabPane.addTab(tabName, geomPiecePanel);
      validate();
      updatePanels();
    } else if (e.getActionCommand() == "del") {
      int index = geomPieceTabPane.getSelectedIndex();
      geomPieceTabPane.removeTabAt(index);
      validate();
      updatePanels();
    }
  }

  //-------------------------------------------------------------------------
  // Add a geometry piece
  //-------------------------------------------------------------------------
  public void addGeomPiece(GeomPiece piece) {
    d_geomPiece.addElement(piece);
  }

  //-------------------------------------------------------------------------
  // Delete a geometry piece
  //-------------------------------------------------------------------------
  public void deleteGeomPieceAt(int index) {
    d_geomPiece.removeElementAt(index);
  }

  //-------------------------------------------------------------------------
  // Update the components
  //-------------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }

  //-------------------------------------------------------------------------
  // Create the geometry pieces from the input particle distribution 
  //-------------------------------------------------------------------------
  public void createPartListGeomPiece(String simComponent) {

    if (d_partList == null) return;

    int numPart = d_partList.size();
    if (!(numPart > 0)) return;
    
    if (d_partGeomPieceExists) return;

    double partThick = ((Particle) d_partList.getParticle(0)).getThickness();

    if (partThick > 0.0) {
      createHollowPartListGeomPiece(simComponent); 
    } else {
      createSolidPartListGeomPiece(simComponent); 
    }

    // Set the exists flag to true
    d_partGeomPieceExists = true;
  }

  //-------------------------------------------------------------------------
  // Delete the geometry pieces for the input particle distribution 
  //-------------------------------------------------------------------------
  public void deletePartListGeomPiece() {

    if (d_partList == null) return;

    int numPart = d_partList.size();
    if (!(numPart > 0)) return;
    
    if (d_partGeomPieceExists) {
      d_geomPiece.removeAllElements();

      // Set the exists flag to false
      d_partGeomPieceExists = false;
    }

  }

  //-------------------------------------------------------------------------
  // Create solid geometry pieces from the input particle distribution 
  //-------------------------------------------------------------------------
  public void createSolidPartListGeomPiece(String simComponent) {

    int partType = ((Particle) d_partList.getParticle(0)).getType();
    if (partType == Particle.CIRCLE) {

      // Get the number of particles and the particle size
      int numPart = d_partList.size();
      double rveSize = d_partList.getRVESize();

      // Get the smallest particle radius and have at least 10 particles
      // in the radial direction
      double minRad = ((Particle) d_partList.getParticle(numPart-1)).getRadius();
      double pointSpacing = minRad/10.0;

      // First add the particles and also create a union of the cylinders
      UnionGeomPiece union = new UnionGeomPiece("all_particles");
      for (int ii = 0; ii < numPart; ++ii) {

        // Get the particle
        Particle part = (Particle) d_partList.getParticle(ii);

        // Get the center, radius, and length
        Point center = part.getCenter();
        double radius = part.getRadius();
        double length = part.getLength();
        if (length == 0.0) {
          length = 0.05*rveSize;
        }
        double thickness = radius;

        // Estimate the number of material points in the radial and axial
        // directions
        int numRadial = (int) Math.ceil(radius/pointSpacing);
        int numAxial = (int) Math.ceil(length/pointSpacing);

        // Find the arc start and end points
        double[] arcPoints = calcArcPoints(center, radius, rveSize);
        double arcStart = arcPoints[0];
        double arcAngle = arcPoints[1];

        // Create a name 
        String name = new String("solid_cylinder_"+String.valueOf(ii));

        // Create a smooth cylinder geometry piece
        SmoothCylGeomPiece piece = 
          new SmoothCylGeomPiece(name, center, radius, thickness, length,
                                 numRadial, numAxial, arcStart, arcAngle); 
        d_geomPiece.addElement(piece);

        // Create a cylinder geometry piece
        String solidName = new String("outer_cylinder_"+String.valueOf(ii));
        CylinderGeomPiece cylPiece = 
          new CylinderGeomPiece(solidName, center, radius, length); 
        union.addGeomPiece(cylPiece);
      }

      // Create a box geometry piece for the domain
      Point min = new Point(0.0,0.0,0.0);
      Point max = new Point(rveSize, rveSize, rveSize);
      BoxGeomPiece box = new BoxGeomPiece("domain", min, max);

      // Create a difference geometry piece for the rest
      DifferenceGeomPiece diff = new DifferenceGeomPiece("rest_of_domain",
                                                         box, union);
      d_geomPiece.addElement(diff);
    }
  }

  //-------------------------------------------------------------------------
  // Create hollow geometry pieces from the input particle distribution 
  //-------------------------------------------------------------------------
  public void createHollowPartListGeomPiece(String simComponent) {

    int partType = ((Particle) d_partList.getParticle(0)).getType();
    if (partType == Particle.CIRCLE) {

      // Get the number of particles and the particle size
      int numPart = d_partList.size();
      double rveSize = d_partList.getRVESize();

      // Get the smallest particle radius and have at least 10 particles
      // in the radial direction
      double minRad = ((Particle) d_partList.getParticle(numPart-1)).getRadius();
      double pointSpacing = minRad/10.0;

      // First add the particles and also create a union of the cylinders
      UnionGeomPiece unionOuter = new UnionGeomPiece("all_particles");
      UnionGeomPiece unionInner = new UnionGeomPiece("all_inside");
      for (int ii = 0; ii < numPart; ++ii) {

        // Get the particle
        Particle part = (Particle) d_partList.getParticle(ii);

        // Get the center, radius, and length
        Point center = part.getCenter();
        double radius = part.getRadius();
        double length = part.getLength();
        if (length == 0.0) {
          length = 0.05*rveSize;
        }
        double thickness = part.getThickness();

        // Estimate the number of material points in the radial and axial
        // directions
        int numRadial = (int) Math.ceil(radius/pointSpacing);
        int numAxial = (int) Math.ceil(length/pointSpacing);

        // Find the arc start and end points
        double[] arcPoints = calcArcPoints(center, radius, rveSize);
        double arcStart = arcPoints[0];
        double arcAngle = arcPoints[1];

        // Create a name 
        String name = new String("hollow_cylinder_"+String.valueOf(ii));

        // Create a smooth cylinder geometry piece
        SmoothCylGeomPiece piece = 
          new SmoothCylGeomPiece(name, center, radius, thickness, length,
                                 numRadial, numAxial, arcStart, arcAngle); 
        d_geomPiece.addElement(piece);

        // Create a cylinder geometry piece for the full cylinder
        String solidName = new String("outer_cylinder_"+String.valueOf(ii));
        CylinderGeomPiece cylPieceSolid = 
          new CylinderGeomPiece(solidName, center, radius, length); 
        unionOuter.addGeomPiece(cylPieceSolid);

        // Create a cylinder geometry piece for the inner hollow region
        // of each cylinder
        String hollowName = new String("inner_cylinder_"+String.valueOf(ii));
        CylinderGeomPiece cylPieceHollow = 
          new CylinderGeomPiece(hollowName, center, radius-thickness, length); 
        unionInner.addGeomPiece(cylPieceHollow);
      }

      // Create a box geometry piece for the domain
      Point min = new Point(0.0,0.0,0.0);
      Point max = new Point(rveSize, rveSize, rveSize);
      BoxGeomPiece box = new BoxGeomPiece("domain", min, max);

      // Create a difference geometry piece for the rest
      DifferenceGeomPiece diff = new DifferenceGeomPiece("rest_of_domain",
                                                         box, unionOuter);
      d_geomPiece.addElement(diff);

      // Add the inner region to the list of geometry pieces
      d_geomPiece.addElement(unionInner);

    }
  }

  //-------------------------------------------------------------------------
  // Calculate intersction of circles with RVE and then calculate arc start
  // and arc angle
  //-------------------------------------------------------------------------
  private double[] calcArcPoints(Point center, double radius, double rveSize) {
    double[] arcPoints = new double[2];
    arcPoints[0] = 0.0;
    arcPoints[1] = 360.0;
    if (intersectsRVE(center, radius, rveSize)) {
      double xmin = center.getX();
      double ymin = center.getY();
      double xmax = rveSize - xmin;
      double ymax = rveSize - ymin;

      // Compute r^2 - (y-y_c)^2 and  r^2 - (x-x_c)^2 for x = 0, rveSize
      // y = 0, rveSize
      double facxmin = radius*radius - xmin*xmin;
      double facymin = radius*radius - ymin*ymin;
      double facxmax = radius*radius - xmax*xmax;
      double facymax = radius*radius - ymax*ymax;

      // No intersection
      if (facxmin < 0.0 && facymin < 0.0 && facxmax < 0.0 && facymax < 0.0) {
        return arcPoints;
      }

      double[] anglexmin = new double[2];
      double[] anglexmax = new double[2];
      double[] angleymin = new double[2];
      double[] angleymax = new double[2];
      for (int ii = 0; ii < 2; ++ii) { 
        anglexmin[ii] = -1.0;
        anglexmax[ii] = -1.0;
        angleymin[ii] = -1.0;
        angleymax[ii] = -1.0;
      }
      double t = 0.0; 

      double xint = 0.0;
      double yint = 0.0;
      int xmincount = 0;
      int xmaxcount = 0;
      int ymincount = 0;
      int ymaxcount = 0;

      // Intersects x = 0
      if (facxmin >= 0.0) {
        t = (ymin - Math.sqrt(facxmin))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = 0.0;
          yint = t*rveSize;
          anglexmin[xmincount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (anglexmin[xmincount] < 0.0) anglexmin[xmincount] += 360.0;
          ++xmincount;
        }
        t = (ymin + Math.sqrt(facxmin))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = 0.0;
          yint = t*rveSize;
          anglexmin[xmincount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (anglexmin[xmincount] < 0.0) anglexmin[xmincount] += 360.0;
          ++xmincount;
        }
      }
      // Intersects x = rveSize
      if (facxmax >= 0.0) {
        t = (ymin - Math.sqrt(facxmax))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = rveSize;
          yint = t*rveSize;
          anglexmax[xmaxcount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (anglexmax[xmaxcount] < 0.0) anglexmax[xmaxcount] += 360.0;
          ++xmaxcount;
        }
        t = (ymin + Math.sqrt(facxmax))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = rveSize;
          yint = t*rveSize;
          anglexmax[xmaxcount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (anglexmax[xmaxcount] < 0.0) anglexmax[xmaxcount] += 360.0;
          ++xmaxcount;
        }
      }
      // Intersects y = 0
      if (facymin >= 0.0) {
        t = (xmin - Math.sqrt(facymin))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = t*rveSize;
          yint = 0.0;
          angleymin[ymincount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (angleymin[ymincount] < 0.0) angleymin[ymincount] += 360.0;
          ++ymincount;
        }
        t = (xmin + Math.sqrt(facymin))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = t*rveSize;
          yint = 0.0;
          angleymin[ymincount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (angleymin[ymincount] < 0.0) angleymin[ymincount] += 360.0;
          ++ymincount;
        }
      }
      // Intersects y = rveSize
      if (facymax >= 0.0) {
        t = (xmin - Math.sqrt(facymax))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = t*rveSize;
          yint = rveSize;
          angleymax[ymaxcount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (angleymax[ymaxcount] < 0.0) angleymax[ymaxcount] += 360.0;
          ++ymaxcount;
        }
        t = (xmin + Math.sqrt(facymax))/rveSize;
        if (t >= 0.0 && t <= 1.0) {
          xint = t*rveSize;
          yint = rveSize;
          angleymax[ymaxcount] = Math.atan2(yint-ymin,xint-xmin)*180.0/Math.PI;
          if (angleymax[ymaxcount] < 0.0) angleymax[ymaxcount] += 360.0;
          ++ymaxcount;
        }
      }

      // Compute arc angle and arc start
      if (xmincount == 1) {
        if (ymincount == 1) {
          arcPoints[0] = angleymin[0];
          arcPoints[1] = anglexmin[0] - angleymin[0];
        } else if (ymaxcount == 1) {
          arcPoints[0] = anglexmin[0];
          arcPoints[1] = angleymax[0] - anglexmin[0];
        }
      } else if (xmincount == 2) {
        arcPoints[0] = anglexmin[0];
        arcPoints[1] = anglexmin[1] + 360.0 - anglexmin[0];
      } else if (xmaxcount == 1) {
        if (ymincount == 1) {
          arcPoints[0] = anglexmax[0];
          arcPoints[1] = angleymin[0] - anglexmax[0];
        } else if (ymaxcount == 1) {
          arcPoints[0] = angleymax[0];
          arcPoints[1] = anglexmax[0] - angleymax[0];
        }
      } else if (xmaxcount == 2) {
        arcPoints[0] = anglexmax[1];
        arcPoints[1] = anglexmax[0] - anglexmax[1];
      } else if (ymincount == 2) {
        if (angleymin[0] < angleymin[1]) {
          arcPoints[0] = angleymin[1];
          arcPoints[1] = 360.0 + angleymin[0] - angleymin[1];
        } else {
          arcPoints[0] = angleymin[1];
          arcPoints[1] = angleymin[0] - angleymin[1];
        }
      } else if (ymaxcount == 2) {
        if (angleymax[0] < angleymax[1]) {
          arcPoints[0] = angleymax[0];
          arcPoints[1] = angleymax[1] - angleymax[0];
        } else {
          arcPoints[0] = angleymax[0];
          arcPoints[1] = 360.0 + angleymax[1] - angleymax[0];
        }
      }


    }
    return arcPoints;
  }

  private boolean intersectsRVE(Point center, double radius, double rveSize) {
    
    double xcen = center.getX();
    double ycen = center.getY();
    double xminCirc = xcen - radius;
    double xmaxCirc = xcen + radius;
    double yminCirc = ycen - radius;
    double ymaxCirc = ycen + radius;

    if ((xminCirc > 0.0) && (yminCirc > 0.0) && (xmaxCirc < rveSize) && 
        (ymaxCirc < rveSize)) return false;
    return true;
  }
}
