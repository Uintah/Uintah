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
import java.util.Random;
import java.util.Vector;
import java.io.*;
import javax.swing.*;

public class BoxGeomPiecePanel extends JPanel 
                               implements ActionListener {
 
  // Data 
  private BoxGeomPiece d_box = null;
  private GeomPiecePanel d_parent = null;

  // Local components
  JTextField nameEntry = null;
  DecimalField xminEntry = null;
  DecimalField yminEntry = null;
  DecimalField zminEntry = null;
  DecimalField xmaxEntry = null;
  DecimalField ymaxEntry = null;
  DecimalField zmaxEntry = null;
  JButton acceptButton = null;

  //--------------------------------------------------------------------
  // Constructor
  //--------------------------------------------------------------------
  public BoxGeomPiecePanel(GeomPiecePanel parent) {

    // Initialize data
    d_box = new BoxGeomPiece();
    d_parent = parent;

    // Set up the GUI
    setLayout(new GridLayout(6,0));

    GridBagLayout gbPanel0 = new GridBagLayout();
    GridBagConstraints gbcPanel0 = new GridBagConstraints();
    JPanel panel0 = new JPanel(gbPanel0);
    JLabel nameLabel = new JLabel("Geom Piece Name");
    UintahGui.setConstraints(gbcPanel0, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gbPanel0.setConstraints(nameLabel, gbcPanel0);
    nameEntry = new JTextField("Box", 20);
    UintahGui.setConstraints(gbcPanel0, GridBagConstraints.NONE,
                             1.0, 1.0, 1, 0, 1, 1, 5);
    gbPanel0.setConstraints(nameEntry, gbcPanel0);
    panel0.add(nameLabel);
    panel0.add(nameEntry);
    add(panel0);

    JLabel minLabel = new JLabel("Lower Left Corner");
    add(minLabel);

    GridBagLayout gbPanel1 = new GridBagLayout();
    GridBagConstraints gbcPanel1 = new GridBagConstraints();
    JPanel panel1 = new JPanel(gbPanel1);
    JLabel xminLabel = new JLabel("x");         
    UintahGui.setConstraints(gbcPanel1, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gbPanel1.setConstraints(xminLabel, gbcPanel1);
    xminEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbcPanel1, GridBagConstraints.NONE,
                             1.0, 1.0, 1, 0, 1, 1, 5);
    gbPanel1.setConstraints(xminEntry, gbcPanel1);
    JLabel yminLabel = new JLabel("y");         
    UintahGui.setConstraints(gbcPanel1, GridBagConstraints.NONE,
                             1.0, 1.0, 2, 0, 1, 1, 5);
    gbPanel1.setConstraints(yminLabel, gbcPanel1);
    yminEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbcPanel1, GridBagConstraints.NONE,
                             1.0, 1.0, 3, 0, 1, 1, 5);
    gbPanel1.setConstraints(yminEntry, gbcPanel1);
    JLabel zminLabel = new JLabel("z");         
    UintahGui.setConstraints(gbcPanel1, GridBagConstraints.NONE,
                             1.0, 1.0, 4, 0, 1, 1, 5);
    gbPanel1.setConstraints(zminLabel, gbcPanel1);
    zminEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbcPanel1, GridBagConstraints.NONE,
                             1.0, 1.0, 5, 0, 1, 1, 5);
    gbPanel1.setConstraints(zminEntry, gbcPanel1);
    panel1.add(xminLabel); panel1.add(xminEntry);
    panel1.add(yminLabel); panel1.add(yminEntry);
    panel1.add(zminLabel); panel1.add(zminEntry);
    add(panel1);

    JLabel maxLabel = new JLabel("Upper Right Corner");
    add(maxLabel);

    GridBagLayout gbPanel2 = new GridBagLayout();
    GridBagConstraints gbcPanel2 = new GridBagConstraints();
    JPanel panel2 = new JPanel(gbPanel2);
    JLabel xmaxLabel = new JLabel("x");         
    UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gbPanel2.setConstraints(xmaxLabel, gbcPanel2);
    xmaxEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE,
                             1.0, 1.0, 1, 0, 1, 1, 5);
    gbPanel2.setConstraints(xmaxEntry, gbcPanel2);
    JLabel ymaxLabel = new JLabel("y");         
    UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE,
                             1.0, 1.0, 2, 0, 1, 1, 5);
    gbPanel2.setConstraints(ymaxLabel, gbcPanel2);
    ymaxEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE,
                             1.0, 1.0, 3, 0, 1, 1, 5);
    gbPanel2.setConstraints(ymaxEntry, gbcPanel2);
    JLabel zmaxLabel = new JLabel("z");         
    UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE,
                             1.0, 1.0, 4, 0, 1, 1, 5);
    gbPanel2.setConstraints(zmaxLabel, gbcPanel2);
    zmaxEntry = new DecimalField(0.0,5);
    UintahGui.setConstraints(gbcPanel2, GridBagConstraints.NONE,
                             1.0, 1.0, 5, 0, 1, 1, 5);
    gbPanel2.setConstraints(zmaxEntry, gbcPanel2);
    panel2.add(xmaxLabel); panel2.add(xmaxEntry);
    panel2.add(ymaxLabel); panel2.add(ymaxEntry);
    panel2.add(zmaxLabel); panel2.add(zmaxEntry);
    add(panel2);

    acceptButton = new JButton("Accept");
    acceptButton.setActionCommand("accept");
    acceptButton.addActionListener(this);
    add(acceptButton);
  }

  //--------------------------------------------------------------------
  // Actions performed when an item is selected
  //--------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {
    if (e.getActionCommand() == "accept") {
      String name = nameEntry.getText();
      double xmin = xminEntry.getValue();
      double ymin = yminEntry.getValue();
      double zmin = zminEntry.getValue();
      double xmax = xmaxEntry.getValue();
      double ymax = ymaxEntry.getValue();
      double zmax = zmaxEntry.getValue();
      Point min = new Point(xmin, ymin, zmin);
      Point max = new Point(xmax, ymax, zmax);
      d_box.set(name, min, max);
      d_parent.addGeomPiece(d_box);
    }
  }
}
