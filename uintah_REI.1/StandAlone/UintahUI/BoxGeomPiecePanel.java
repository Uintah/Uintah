//**************************************************************************
// Class   : BoxGeomPiecePanel
// Purpose : Panel to enter data for box geometry pieces
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class BoxGeomPiecePanel extends JPanel 
                               implements ActionListener {
 
  // Data 
  private BoxGeomPiece d_box = null;
  private GeomPiecePanel d_parent = null;

  // Local components
  JTextField nameEntry = null;
  DecimalVectorField minEntry = null;
  DecimalVectorField maxEntry = null;
  JButton acceptButton = null;

  //--------------------------------------------------------------------
  // Constructor
  //--------------------------------------------------------------------
  public BoxGeomPiecePanel(GeomPiecePanel parent) {

    // Initialize data
    d_box = new BoxGeomPiece();
    d_parent = parent;

    // Create a grid bag for the components
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);
    int fill = GridBagConstraints.BOTH;
    int xgap = 5;
    int ygap = 0;

    JLabel nameLabel = new JLabel("Geom Piece Name");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 0);
    gb.setConstraints(nameLabel, gbc);
    add(nameLabel);

    nameEntry = new JTextField("Box", 20);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 0);
    gb.setConstraints(nameEntry, gbc);
    add(nameEntry);

    JLabel minLabel = new JLabel("Lower Left Corner");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 1);
    gb.setConstraints(minLabel, gbc);
    add(minLabel);

    minEntry = new DecimalVectorField(0.0, 0.0, 0.0, 5, true);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 1);
    gb.setConstraints(minEntry, gbc);
    add(minEntry);

    JLabel maxLabel = new JLabel("Upper Right Corner");
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 2);
    gb.setConstraints(maxLabel, gbc);
    add(maxLabel);

    maxEntry = new DecimalVectorField(0.0, 0.0, 0.0, 5, true);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 2);
    gb.setConstraints(maxEntry, gbc);
    add(maxEntry);

    acceptButton = new JButton("Accept");
    acceptButton.setActionCommand("accept");
    acceptButton.addActionListener(this);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 3);
    gb.setConstraints(acceptButton, gbc);
    add(acceptButton);
  }

  //--------------------------------------------------------------------
  // Actions performed when an item is selected
  //--------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {
    if (e.getActionCommand() == "accept") {
      String name = nameEntry.getText();
      double xmin = minEntry.x();
      double ymin = minEntry.y();
      double zmin = minEntry.z();
      double xmax = maxEntry.x();
      double ymax = maxEntry.y();
      double zmax = maxEntry.z();
      Point min = new Point(xmin, ymin, zmin);
      Point max = new Point(xmax, ymax, zmax);
      d_box.set(name, min, max);
      d_parent.addGeomPiece(d_box);
    }
  }
}
