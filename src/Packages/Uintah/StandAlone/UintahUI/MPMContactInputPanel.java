//**************************************************************************
// Class   : MPMContactInputPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           contact between MPM materials
// Author  : Biswajit Banerjee
// Date    : 05/17/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.event.*;
import java.util.Vector;

public class MPMContactInputPanel extends JPanel 
                                  implements ItemListener {

  // Data and components
  private String d_contactType = null;
  private Vector d_mpmMat = null;

  private JComboBox contactTypeComB = null;
  private JList contactMatList = null;
  private JScrollPane contactMatSP = null;
  private DecimalField frictionEntry = null;
  private JComboBox contactDirComB = null;

  public MPMContactInputPanel(Vector mpmMat) {

    // Initialize
    d_contactType = new String("rigid");
    d_mpmMat = mpmMat;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    JLabel contactTypeLabel = new JLabel("Contact Type");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 0, 0);
    gb.setConstraints(contactTypeLabel, gbc);
    add(contactTypeLabel); 

    contactTypeComB = new JComboBox();
    contactTypeComB.addItem("No Contact");
    contactTypeComB.addItem("Rigid Contact");
    contactTypeComB.addItem("Specified Velocity Contact");
    contactTypeComB.addItem("Single Velocity Contact");
    contactTypeComB.addItem("Approach Contact");
    contactTypeComB.addItem("Frictional Contact");
    contactTypeComB.addItemListener(this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 1, 0);
    gb.setConstraints(contactTypeComB, gbc);
    add(contactTypeComB); 
    
    JLabel contactMatLabel = new JLabel("Contact Materials");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 0, 1);
    gb.setConstraints(contactMatLabel, gbc);
    add(contactMatLabel); 

    contactMatList = new JList(d_mpmMat);
    contactMatSP = new JScrollPane(contactMatList);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 1, 1);
    gb.setConstraints(contactMatSP, gbc);
    add(contactMatSP); 
    
    JLabel frictionLabel = new JLabel("Friction Coefficient");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 0, 2);
    gb.setConstraints(frictionLabel, gbc);
    add(frictionLabel); 

    frictionEntry = new DecimalField(0.0, 7);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 1, 2);
    gb.setConstraints(frictionEntry, gbc);
    add(frictionEntry); 

    JLabel contactDirLabel = new JLabel("Initial Contact Direction");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 0, 3);
    gb.setConstraints(contactDirLabel, gbc);
    add(contactDirLabel); 

    contactDirComB = new JComboBox();
    contactDirComB.addItem("X-Direction");
    contactDirComB.addItem("Y-Direction");
    contactDirComB.addItem("Z-Direction");
    contactDirComB.addItemListener(this);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 1, 3);
    gb.setConstraints(contactDirComB, gbc);
    add(contactDirComB); 

  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Listens for item picked in combo box and takes action as required.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  public void itemStateChanged(ItemEvent e) {
        
    // Get the object that has been selected
    Object source = e.getItemSelectable();

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());

    if (source == contactTypeComB) {
    } else {

    }
  }

  //--------------------------------------------------------------------
  /** Write the contents out in Uintah format */
  //--------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
      
    if (pw == null) return;


  }
}
