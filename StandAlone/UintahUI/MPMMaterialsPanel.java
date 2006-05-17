/**************************************************************************
// Program : MPMMaterialsPanel.java
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
import javax.swing.text.*;
import javax.swing.event.*;
import java.text.DecimalFormat;
import java.util.Vector;
import java.awt.Point;

//**************************************************************************
// Class   : MPMMaterialsPanel
//**************************************************************************
public class MPMMaterialsPanel extends JPanel {

  // Static variables

  // Data
  private Vector d_mpmMat = null;
  private UintahInputPanel d_parent = null;

  // Local components
  private JTabbedPane mpmMatTabbedPane = null;
  private Vector mpmMatInputPanel = null;
  private JButton addButton = null;
  private JButton delButton = null;
  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public MPMMaterialsPanel(Vector mpmMat, 
                           UintahInputPanel parent) {

    // Initialize local variables
    d_mpmMat = mpmMat;
    d_parent = parent;

    // Initialize the material input panel vector
    mpmMatInputPanel = new Vector();

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Create the panels for each material
    String matID = new String("Material "+String.valueOf(0));
    d_mpmMat.addElement(matID);

    MPMMaterialInputPanel matPanel = new MPMMaterialInputPanel(0);
    mpmMatInputPanel.addElement(matPanel);

    // Create the tabbed pane
    mpmMatTabbedPane = new JTabbedPane();
    mpmMatTabbedPane.addTab(matID, null, matPanel, null);
    mpmMatTabbedPane.setSelectedIndex(0);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 0, 
                             GridBagConstraints.REMAINDER, 1, 5);
    gb.setConstraints(mpmMatTabbedPane, gbc);
    add(mpmMatTabbedPane);

    // Create the add button
    addButton = new JButton("Add Material");
    addButton.setActionCommand("add");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(addButton, gbc);
    add(addButton);

    // Create the delete button
    delButton = new JButton("Remove Material");
    delButton.setActionCommand("delete");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 1, 1, 1, 1, 5);
    gb.setConstraints(delButton, gbc);
    add(delButton);

    // Create the save button
    saveButton = new JButton("Save");
    saveButton.setActionCommand("save");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 2, 1, 1, 1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    addButton.addActionListener(buttonListener);
    delButton.addActionListener(buttonListener);
    saveButton.addActionListener(buttonListener);
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

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "add") {

        int ii = d_mpmMat.size();
        String matID = new String("Material "+String.valueOf(ii));
        d_mpmMat.addElement(matID);

        MPMMaterialInputPanel matPanel = new MPMMaterialInputPanel(ii);
        mpmMatInputPanel.addElement(matPanel);
        mpmMatTabbedPane.addTab(matID, null, matPanel, null);
        mpmMatTabbedPane.setSelectedIndex(ii);

      } else if (e.getActionCommand() == "delete") {

        int ii = d_mpmMat.size()-1;
        d_mpmMat.removeElementAt(ii);

        mpmMatInputPanel.removeElementAt(ii);
        mpmMatTabbedPane.remove(ii);
        mpmMatTabbedPane.setSelectedIndex(ii);

      } else if (e.getActionCommand() == "save") {
        
        // Create filewriter and printwriter
        File outputFile = new File("test.ups");
        try {
          FileWriter fw = new FileWriter(outputFile);
          PrintWriter pw = new PrintWriter(fw);

          String tab = new String("    ");
          for (int ii = 0; ii < d_mpmMat.size(); ++ii) {
            MPMMaterialInputPanel matPanel = 
              (MPMMaterialInputPanel) mpmMatInputPanel.elementAt(ii);
            matPanel.writeUintah(pw, tab);
          }

          pw.close();
          fw.close();
        } catch (Exception event) {
          System.out.println("Could not write to file "+outputFile.getName());
        }
      }
    }
  }


}
