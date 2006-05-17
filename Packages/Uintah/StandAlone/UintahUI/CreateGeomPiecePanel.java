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
import java.util.Random;
import java.util.Vector;
import java.io.*;
import javax.swing.*;

public class CreateGeomPiecePanel extends JPanel 
                                  implements ActionListener {

  // Data
  private boolean d_usePartDist = false;
  private InputGeometryPanel d_parent = null;
  private Vector d_geomPiece = null;
  private ParticleList d_partList = null;

  // Components
  private JButton addButton = null;
  private JButton delButton = null;
  private JTabbedPane geomPieceTabPane = null;

  //-------------------------------------------------------------------------
  // Constructor
  //-------------------------------------------------------------------------
  public CreateGeomPiecePanel(boolean usePartDist,
                              ParticleList partList,
                              Vector geomPiece,
                              InputGeometryPanel parent) {

    // Initialize
    d_usePartDist = usePartDist;
    d_geomPiece = geomPiece;
    d_partList = partList;

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

  //---------------------------------------------------------------------
  // Update the usePartDist flag
  //---------------------------------------------------------------------
  public void usePartDist(boolean flag) {
    d_usePartDist = flag;
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
  public void createPartDistGeomPiece() {

    if (d_partList == null) return;

    int numPart = d_partList.size();
    if (!(numPart > 0)) return;
    
    int partType = ((Particle) d_partList.getParticle(0)).getType();

    if (partType == Particle.CIRCLE) {

      // First add the particles and also create a union of the cylinders
      UnionGeomPiece union = new UnionGeomPiece("all_particles");
      for (int ii = 0; ii < numPart; ++ii) {

        // Get the particle
        Particle part = (Particle) d_partList.getParticle(ii);

        // Get the center, radius, and length
        Point center = part.getCenter();
        double radius = part.getRadius();
        double length = part.getLength();

        // Create a name 
        String name = new String("cylinder_"+String.valueOf(ii));

        // Create a smooth cylinder geometry piece
        SmoothCylGeomPiece piece = 
          new SmoothCylGeomPiece(name, center, radius, length); 
        d_geomPiece.addElement(piece);

        // Create a cylinder geometry piece
        CylinderGeomPiece cylPiece = 
          new CylinderGeomPiece(name, center, radius, length); 
        union.addGeomPiece(cylPiece);
      }
      d_geomPiece.addElement(union);

      // Create a box geometry piece for the domain
      double rveSize = d_partList.getRVESize();
      Point min = new Point(0.0,0.0,0.0);
      Point max = new Point(rveSize, rveSize, rveSize);
      BoxGeomPiece box = new BoxGeomPiece("domain", min, max);

      // Create a difference geometry piece for the rest
      DifferenceGeomPiece diff = new DifferenceGeomPiece("rest_of_domain",
         box, union);
      d_geomPiece.addElement(diff);
    }
  }


}
