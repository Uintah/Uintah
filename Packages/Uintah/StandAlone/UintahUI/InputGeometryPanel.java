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
  private boolean d_usePartDist = false;

  private ParticleList d_partList = null;
  private GeometryPanel d_parent = null;
 
  private Vector d_mpmMat = null;
  private Vector d_iceMat = null;
  private Vector d_geomObj = null;
  private Vector d_geomPiece = null;
  
  // Local components
  private JCheckBox usePartDistCB = null;
  private JButton saveButton = null;
  private CreateGeomObjectPanel createGeomObjectPanel = null;
  private CreateGeomPiecePanel createGeomPiecePanel = null;

  //-------------------------------------------------------------------------
  // Construct an input panel for the geometry
  //-------------------------------------------------------------------------
  public InputGeometryPanel(ParticleList partList,
                            Vector mpmMat,
                            Vector iceMat,
                            Vector geomObj,
                            Vector geomPiece,
                            GeometryPanel parent) {

    // Initialize
    d_domainSize = 100.0;
    d_usePartDist = false;

    // Save the input arguments
    d_partList = partList;
    d_mpmMat = mpmMat;
    d_iceMat = iceMat;
    d_geomObj = geomObj;
    d_geomPiece = geomPiece;
    d_parent = parent;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create check box
    usePartDistCB = new JCheckBox("Use Computed Particle Distribution");
    usePartDistCB.setSelected(false);
    usePartDistCB.addItemListener(this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(usePartDistCB, gbc);
    add(usePartDistCB);

    // Create a panel for creating geometry pieces
    createGeomPiecePanel = 
      new CreateGeomPiecePanel(d_usePartDist, d_partList, d_geomPiece, this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(createGeomPiecePanel, gbc);
    add(createGeomPiecePanel);

    // Create a panel for creating geometry objects
    createGeomObjectPanel = 
      new CreateGeomObjectPanel(d_usePartDist, d_partList, d_mpmMat, d_iceMat,
                                d_geomObj, d_geomPiece, this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 2, 1, 1, 5);
    gb.setConstraints(createGeomObjectPanel, gbc);
    add(createGeomObjectPanel);

    // Create save button
    saveButton = new JButton("Save Calculated Data");
    saveButton.setActionCommand("save");
    saveButton.addActionListener(this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE,
                             1.0, 1.0, 0, 3, 1, 1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

  }

  //-------------------------------------------------------------------------
  // Update the components
  //-------------------------------------------------------------------------
  public void updatePanels() {
    d_parent.updatePanels();
  }

  //-------------------------------------------------------------------------
  // Actions performed when check box state changes
  //-------------------------------------------------------------------------
  public void itemStateChanged(ItemEvent e) {

    // Get the object that has changed
    Object source = e.getItemSelectable();

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());

    if (source == usePartDistCB) {
      if (e.getStateChange() == ItemEvent.SELECTED) {
        d_usePartDist = true;
        createGeomObjectPanel.usePartDist(d_usePartDist);
        createGeomPiecePanel.usePartDist(d_usePartDist);
        createGeomPiecePanel.setVisible(false);

        if (d_partList != null) {
          d_domainSize = d_partList.getRVESize();
          createGeomPiecePanel.createPartDistGeomPiece();
          refreshDisplay();
        }

      } else {
        d_usePartDist = false;
        createGeomObjectPanel.usePartDist(d_usePartDist);
        createGeomPiecePanel.usePartDist(d_usePartDist);
        createGeomPiecePanel.setVisible(true);
      }
    }
  }

  //-------------------------------------------------------------------------
  // Actions performed when a button is pressed
  //-------------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    if (e.getActionCommand() == "save") {

      // Create filewriter and printwriter
      File outputFile = new File("test.ups");
      try {
        FileWriter fw = new FileWriter(outputFile);
        PrintWriter pw = new PrintWriter(fw);

        String tab = new String("    ");
        writeUintah(pw, tab);

        pw.close();
        fw.close();
      } catch (Exception event) {
        System.out.println("Could not write to file "+outputFile.getName());
      }
    }
  }

  //-------------------------------------------------------------------------
  // Refresh display
  //-------------------------------------------------------------------------
  public void refreshDisplay() {
    d_parent.setDomainSize(d_domainSize);
    d_parent.refreshDisplayGeometryPanel();
  }

  //-------------------------------------------------------------------------
  // Write in Uintah format
  //-------------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
  }
  public void print() {
  }



}
