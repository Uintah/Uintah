//**************************************************************************
// Class   : GeomPiecePanel
// Purpose : Panel to enter data on geometry pieces
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class GeomPiecePanel extends JPanel 
                            implements ActionListener {

  // Data
  private CreateGeomPiecePanel d_parent = null;

  // Components
  private JComboBox geomComB = null;
  private GridBagLayout gb = null;
  private GridBagConstraints gbc = null;

  //-------------------------------------------------------------------------
  // Constructor
  //-------------------------------------------------------------------------
  public GeomPiecePanel(CreateGeomPiecePanel parent) {

    // Save parent
    d_parent = parent;

    // Create a grid bag for the components
    gb = new GridBagLayout();
    gbc = new GridBagConstraints();
    setLayout(gb);

    // Create a combo box for choosing the geometry piece type
    JLabel geomLabel = new JLabel("Geometry Piece Type");
    UintahGui.setConstraints(gbc, 0, 0);
    gb.setConstraints(geomLabel, gbc);
    add(geomLabel);
    geomComB = new JComboBox();
    geomComB.addItem("Box");
    geomComB.addItem("Cylinder");
    geomComB.addItem("Sphere");
    geomComB.addItem("Cone");
    geomComB.addActionListener(this);
     
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
			     1.0, 1.0, 1, 0, 1, 1, 5);
    gb.setConstraints(geomComB, gbc);
    add(geomComB);
  }

  //-------------------------------------------------------------------------
  // Actions performed when an item is selected
  //-------------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    // Find the object that has been selected
    JComboBox source = (JComboBox) e.getSource();

    // Get the item that has been selected
    String item = (String) source.getSelectedItem();

    if (item.equals(new String("Box"))) {
      BoxGeomPiecePanel boxPanel = new BoxGeomPiecePanel(this);
      UintahGui.setConstraints(gbc, GridBagConstraints.BOTH,
			       1.0, 1.0, 0, 1, 1, 1, 5);
      gb.setConstraints(boxPanel, gbc);
      add(boxPanel);
      validate();
      updatePanels();
    } else if (item.equals(new String("Cylinder"))) {
    } else if (item.equals(new String("Sphere"))) {
    } else if (item.equals(new String("Cone"))) {
    } 

  }

  //-------------------------------------------------------------------------
  // Update the components
  //-------------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }

  //-------------------------------------------------------------------------
  // Add a geometry piece
  //-------------------------------------------------------------------------
  public void addGeomPiece(GeomPiece piece) {
    d_parent.addGeomPiece(piece);
  }

}
