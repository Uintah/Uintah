//**************************************************************************
// Program : IntegerVectorField.java
// Purpose : An extension of JPanel to take integer vectors as input
// Author  : Biswajit Banerjee
// Date    : 05/20/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import javax.swing.*;

public class IntegerVectorField extends JPanel {

  // Data
  private JLabel xLabel = null;
  private JLabel yLabel = null;
  private JLabel zLabel = null;
  private IntegerField xEntry = null;
  private IntegerField yEntry = null;
  private IntegerField zEntry = null;

  // Constructor
  public IntegerVectorField(int xval, int yval, int zval, int columns) {

    // Create a grid bag for the components
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    xLabel = new JLabel("x");         
    setConstraints(gbc, 0, 0);
    gb.setConstraints(xLabel, gbc);
    add(xLabel); 

    xEntry = new IntegerField(xval, columns);
    setConstraints(gbc, 1, 0);
    gb.setConstraints(xEntry, gbc);
    add(xEntry);

    yLabel = new JLabel("y");         
    setConstraints(gbc, 2, 0);
    gb.setConstraints(yLabel, gbc);
    add(yLabel); 

    yEntry = new IntegerField(yval, columns);
    setConstraints(gbc, 3, 0);
    gb.setConstraints(yEntry, gbc);
    add(yEntry);

    zLabel = new JLabel("z");         
    setConstraints(gbc, 4, 0);
    gb.setConstraints(zLabel, gbc);
    add(zLabel); 

    zEntry = new IntegerField(zval, columns);
    setConstraints(gbc, 5, 0);
    gb.setConstraints(zEntry, gbc);
    add(zEntry);
    
  }

  // Get values from the fields
  public int x() {
    return xEntry.getValue();
  }

  // Get values from the fields
  public int y() {
    return yEntry.getValue();
  }

  // Get values from the fields
  public int z() {
    return zEntry.getValue();
  }

  // Enable/disable the field
  public void setEnabled(boolean enable) {
    if (enable) {
      xLabel.setEnabled(true);
      yLabel.setEnabled(true);
      zLabel.setEnabled(true);
      xEntry.setEnabled(true);
      yEntry.setEnabled(true);
      zEntry.setEnabled(true);
    } else {
      xLabel.setEnabled(false);
      yLabel.setEnabled(false);
      zLabel.setEnabled(false);
      xEntry.setEnabled(false);
      yEntry.setEnabled(false);
      zEntry.setEnabled(false);
    }
  }

  // For setting the gridbagconstraints for this object
  private void setConstraints(GridBagConstraints c, int col, int row) {
    c.fill = GridBagConstraints.NONE;
    c.weightx = 1.0;
    c.weighty = 1.0;
    c.gridx = col;
    c.gridy = row;
    c.gridwidth = 1;
    c.gridheight = 1;
    Insets insets = new Insets(5, 5, 5, 5);
    c.insets = insets;
  }

}
