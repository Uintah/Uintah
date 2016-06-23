/**************************************************************************
// Class   : MPMMaterialsPanel
// Purpose : Create a panel that contains widgets to take inputs for
//           MPM materials
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************/

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.Vector;

public class MPMMaterialsPanel extends JPanel 
                               implements ChangeListener {

  // Static variables

  // Data
  private Vector d_geomObj = null;
  private Vector d_mpmMat = null;

  // Local components
  private JTabbedPane mpmMatTabbedPane = null;
  private Vector mpmMatInputPanel = null;
  private JButton addButton = null;
  private JButton delButton = null;

  private MPMContactInputPanel contactPanel = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public MPMMaterialsPanel(Vector geomObj,
                           Vector mpmMat, 
                           UintahInputPanel parent) {

    // Initialize local variables
    d_geomObj = geomObj;
    d_mpmMat = mpmMat;

    // Initialize the material input panel vector
    mpmMatInputPanel = new Vector();

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the add button
    addButton = new JButton("Add Material");
    addButton.setActionCommand("add");
    UintahGui.setConstraints(gbc, 0, 0);
    gb.setConstraints(addButton, gbc);
    add(addButton);

    // Create the delete button
    delButton = new JButton("Remove Material");
    delButton.setActionCommand("delete");
    UintahGui.setConstraints(gbc, 1, 0);
    gb.setConstraints(delButton, gbc);
    add(delButton);

    // Create a panel for the first material
    String matID = new String("MPM Material "+String.valueOf(0));
    d_mpmMat.addElement(matID);

    MPMMaterialInputPanel matPanel = new MPMMaterialInputPanel(0, d_geomObj);
    mpmMatInputPanel.addElement(matPanel);
    
    contactPanel = new MPMContactInputPanel(d_mpmMat);

    // Create the tabbed pane
    mpmMatTabbedPane = new JTabbedPane();
    mpmMatTabbedPane.addTab(matID, null, matPanel, null);
    mpmMatTabbedPane.addTab("Contact", null, contactPanel, null);
    mpmMatTabbedPane.setSelectedIndex(0);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 
                             GridBagConstraints.REMAINDER, 1, 5);
    gb.setConstraints(mpmMatTabbedPane, gbc);
    add(mpmMatTabbedPane);
    mpmMatTabbedPane.addChangeListener(this);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    addButton.addActionListener(buttonListener);
    delButton.addActionListener(buttonListener);
  }

  //-----------------------------------------------------------------------
  // Actions when a tab is selected
  //-----------------------------------------------------------------------
  public void stateChanged(ChangeEvent e) {

    // Get the number of tabs and the selected index
    int numTab = mpmMatTabbedPane.getTabCount();
    int tabIndex = mpmMatTabbedPane.getSelectedIndex();
    if (tabIndex == numTab - 1) {
      contactPanel.refresh();
    }
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
    int numMat = d_mpmMat.size();
    for (int ii = 0; ii < numMat; ++ii) {
      MPMMaterialInputPanel matPanel = 
        (MPMMaterialInputPanel) mpmMatInputPanel.elementAt(ii);
      matPanel.refresh();
    }
  }

  //-----------------------------------------------------------------------
  // Create MPM materials for particle distribution
  //-----------------------------------------------------------------------
  public void createPartListMPMMaterial(String simType) {

    if (simType.equals(new String("mpm"))) {
      int numMat = d_mpmMat.size();
      if (numMat < 2) {
        String matID = new String("Material "+String.valueOf(numMat));
        d_mpmMat.addElement(matID);

        MPMMaterialInputPanel matPanel = 
          new MPMMaterialInputPanel(numMat, d_geomObj);
        mpmMatInputPanel.addElement(matPanel);
        mpmMatTabbedPane.add(matPanel, numMat);
        mpmMatTabbedPane.setTitleAt(numMat, matID);
        mpmMatTabbedPane.setSelectedIndex(0);
      }
    }
  }

  //---------------------------------------------------------------
  // Write out in Uintah format
  //---------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab, int matIndex) {

    if (pw == null) return;

    int numMat = d_mpmMat.size();
    if (matIndex < numMat) {
      MPMMaterialInputPanel matPanel = 
        (MPMMaterialInputPanel) mpmMatInputPanel.elementAt(matIndex);
      matPanel.writeUintah(pw, tab);
    }
  }

  public void writeUintahContact(PrintWriter pw, String tab) {

    if (pw == null) return;

    contactPanel.writeUintah(pw, tab);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "add") {

        int ii = d_mpmMat.size();
        String matID = new String("Material "+String.valueOf(ii));
        d_mpmMat.addElement(matID);

        MPMMaterialInputPanel matPanel = 
          new MPMMaterialInputPanel(ii, d_geomObj);
        mpmMatInputPanel.addElement(matPanel);
        mpmMatTabbedPane.add(matPanel, ii);
        mpmMatTabbedPane.setTitleAt(ii,matID);
        mpmMatTabbedPane.setSelectedIndex(ii);

      } else if (e.getActionCommand() == "delete") {

        int ii = d_mpmMat.size()-1;
        d_mpmMat.removeElementAt(ii);

        mpmMatInputPanel.removeElementAt(ii);
        mpmMatTabbedPane.remove(ii);
        mpmMatTabbedPane.setSelectedIndex(ii);

      }
    }
  }


}
