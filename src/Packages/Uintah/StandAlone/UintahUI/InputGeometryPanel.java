//**************************************************************************
// Class   : InputGeometryPanel.java
// Purpose : Create a panel to take geometry inputs.
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.awt.event.*;
import java.util.Random;
import java.util.Vector;
import java.io.PrintWriter;
import java.io.*;
import javax.swing.*;

public class InputGeometryPanel extends JPanel 
                                implements ItemListener,
                                           ActionListener {

  // Local data
  private double d_domainSize = 0.0;
  private boolean d_usePartList = false;

  private ParticleList d_partList = null;
  private GeometryPanel d_parent = null;
 
  private Vector d_geomObj = null;
  private Vector d_geomPiece = null;
  
  // Local components
  private JCheckBox usePartListCB = null;
  private CreateGeomObjectPanel createGeomObjectPanel = null;
  private CreateGeomPiecePanel createGeomPiecePanel = null;

  //-------------------------------------------------------------------------
  // Construct an input panel for the geometry
  //-------------------------------------------------------------------------
  public InputGeometryPanel(ParticleList partList,
                            Vector geomObj,
                            Vector geomPiece,
                            GeometryPanel parent) {

    // Initialize
    d_domainSize = 100.0;
    d_usePartList = false;

    // Save the input arguments
    d_partList = partList;
    d_geomObj = geomObj;
    d_geomPiece = geomPiece;
    d_parent = parent;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create check box
    usePartListCB = new JCheckBox("Use Computed Particle Distribution");
    usePartListCB.setSelected(false);
    usePartListCB.addItemListener(this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(usePartListCB, gbc);
    add(usePartListCB);

    // Create a panel for creating geometry pieces
    createGeomPiecePanel = 
      new CreateGeomPiecePanel(d_usePartList, d_partList, 
                               d_geomPiece, this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(createGeomPiecePanel, gbc);
    add(createGeomPiecePanel);

    // Create a panel for creating geometry objects
    createGeomObjectPanel = 
      new CreateGeomObjectPanel(d_usePartList, d_partList, 
                                d_geomObj, d_geomPiece, this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 2, 1, 1, 5);
    gb.setConstraints(createGeomObjectPanel, gbc);
    add(createGeomObjectPanel);
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {

    if (d_partList == null) return;

    if (d_partList.size() > 0) {
      d_usePartList = true;
      d_domainSize = d_partList.getRVESize();
      updatePanels();
    }
  }

  //-------------------------------------------------------------------------
  // Update the components
  //-------------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }

  //-------------------------------------------------------------------------
  // Update geometry objects
  //-------------------------------------------------------------------------
  public void createPartListGeomObjects() {
    d_usePartList = true;

    // Don't allow the creation of any more geometry pieces
    createGeomPiecePanel.usePartList(d_usePartList);
    createGeomPiecePanel.setEnabled(false);

    // Set up the geometry object panel
    createGeomObjectPanel.usePartList(d_usePartList);
    createGeomObjectPanel.disableCreate();
    createGeomObjectPanel.disableDelete();
    if (d_partList != null) {
      d_domainSize = d_partList.getRVESize();
      createGeomPiecePanel.createPartListGeomPiece();
      createGeomObjectPanel.addPartListGeomObjectPanel();
      refreshDisplay();
    }
  }

  //-------------------------------------------------------------------------
  // Actions performed when check box state changes
  //-------------------------------------------------------------------------
  public void itemStateChanged(ItemEvent e) {

    // Get the object that has changed
    Object source = e.getItemSelectable();

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());

    if (source == usePartListCB) {
      if (e.getStateChange() == ItemEvent.SELECTED) {
        d_usePartList = true;
        createGeomObjectPanel.usePartList(d_usePartList);
        createGeomPiecePanel.usePartList(d_usePartList);
        createGeomPiecePanel.setVisible(false);

        if (d_partList != null) {
          d_domainSize = d_partList.getRVESize();
          createGeomPiecePanel.createPartListGeomPiece();
          refreshDisplay();
        }

      } else {
        d_usePartList = false;
        createGeomObjectPanel.usePartList(d_usePartList);
        createGeomPiecePanel.usePartList(d_usePartList);
        createGeomPiecePanel.setVisible(true);
      }
    }
  }

  //-------------------------------------------------------------------------
  // Actions performed when a button is pressed
  //-------------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

  }

  //-------------------------------------------------------------------------
  // Refresh display
  //-------------------------------------------------------------------------
  public void refreshDisplay() {
    d_parent.setDomainSize(d_domainSize);
    d_parent.refreshDisplayGeometryPanel();
  }

}
