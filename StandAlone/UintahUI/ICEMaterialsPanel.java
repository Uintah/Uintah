/**************************************************************************
// Class   : ICEMaterialsPanel
// Purpose : Create a panel that contains widgets to take inputs for
//           ICE materials
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************/

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import java.util.Vector;

public class ICEMaterialsPanel extends JPanel 
                               implements ActionListener {

  // Static variables

  // Data
  private Vector d_geomObj = null;
  private Vector d_iceMat = null;

  // Local components
  private JTabbedPane iceMatTabbedPane = null;
  private Vector iceMatInputPanel = null;
  private JButton addButton = null;
  private JButton delButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public ICEMaterialsPanel(Vector geomObj,
                           Vector iceMat,  
                           UintahInputPanel parent) {

    // Initialize local variables
    d_geomObj = geomObj;
    d_iceMat = iceMat;

    // Initialice the input panel vector
    iceMatInputPanel = new Vector();

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the add button
    addButton = new JButton("Add Material");
    addButton.setActionCommand("add");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(addButton, gbc);
    add(addButton);

    // Create the delete button
    delButton = new JButton("Remove Material");
    delButton.setActionCommand("delete");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 1, 0, 1, 1, 5);
    gb.setConstraints(delButton, gbc);
    add(delButton);

    // Create a panel for the first material
    String matID = new String("ICE Material "+String.valueOf(0));
    d_iceMat.addElement(matID);

    ICEMaterialInputPanel matPanel = new ICEMaterialInputPanel(0, d_geomObj);
    iceMatInputPanel.addElement(matPanel);

    // Create the tabbed pane
    iceMatTabbedPane = new JTabbedPane();
    iceMatTabbedPane.addTab(matID, null, matPanel, null);
    iceMatTabbedPane.setSelectedIndex(0);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 
                             GridBagConstraints.REMAINDER, 1, 5);
    gb.setConstraints(iceMatTabbedPane, gbc);
    add(iceMatTabbedPane);

    // Add listener
    addButton.addActionListener(this);
    delButton.addActionListener(this);
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
    int numMat = d_iceMat.size();
    for (int ii = 0; ii < numMat; ++ii) {
      ICEMaterialInputPanel matPanel = 
        (ICEMaterialInputPanel) iceMatInputPanel.elementAt(ii);
      matPanel.refresh();
    }
  }

  //-----------------------------------------------------------------------
  // Create ICE materials for particle distribution
  //-----------------------------------------------------------------------
  public void createPartListICEMaterial(String simType) {

    if (simType.equals(new String("mpmice"))) {
      int numMat = d_iceMat.size();
      if (numMat < 2) {
        String matID = new String("ICE Material "+String.valueOf(numMat));
        d_iceMat.addElement(matID);

        ICEMaterialInputPanel matPanel = 
          new ICEMaterialInputPanel(numMat, d_geomObj);
        iceMatInputPanel.addElement(matPanel);
        iceMatTabbedPane.add(matPanel, numMat);
        iceMatTabbedPane.setTitleAt(numMat, matID);
        iceMatTabbedPane.setSelectedIndex(0);
      }
    }
  }

  //---------------------------------------------------------------
  // Write out in Uintah format
  //---------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab, int matIndex) {

    if (pw == null) return;

    int numMat = d_iceMat.size();
    if (matIndex < numMat) {
      ICEMaterialInputPanel matPanel = 
        (ICEMaterialInputPanel) iceMatInputPanel.elementAt(matIndex);
      matPanel.writeUintah(pw, tab);
    }
  }

  //---------------------------------------------------------------
  // Respond to button pressed 
  //---------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {

    if (e.getActionCommand() == "add") {

      int ii = d_iceMat.size();
      String matID = new String("ICE Material "+String.valueOf(ii));
      d_iceMat.addElement(matID);

      ICEMaterialInputPanel matPanel = 
        new ICEMaterialInputPanel(ii, d_geomObj);
      iceMatInputPanel.addElement(matPanel);
      iceMatTabbedPane.add(matPanel, ii);
      iceMatTabbedPane.setTitleAt(ii,matID);
      iceMatTabbedPane.setSelectedIndex(ii);

    } else if (e.getActionCommand() == "delete") {

      int ii = d_iceMat.size()-1;
      d_iceMat.removeElementAt(ii);

      iceMatInputPanel.removeElementAt(ii);
      iceMatTabbedPane.remove(ii);
      iceMatTabbedPane.setSelectedIndex(ii);

    }
  }

}
