//**************************************************************************
// Program : RigidMaterialPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           rigid materials
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.io.*;
import javax.swing.*;

public class RigidMaterialPanel extends JPanel {

  // Data and components
  private DecimalField bulkEntry = null;
  private DecimalField shearEntry = null;
  private DecimalField cteEntry = null;

  public RigidMaterialPanel() {

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    JLabel bulkLabel = new JLabel("Bulk Modulus");
    UintahGui.setConstraints(gbc, 0, 0);
    gb.setConstraints(bulkLabel, gbc);
    add(bulkLabel);

    bulkEntry = new DecimalField(130.0e9, 9, true);
    UintahGui.setConstraints(gbc, 1, 0);
    gb.setConstraints(bulkEntry, gbc);
    add(bulkEntry);

    JLabel shearLabel = new JLabel("Shear Modulus");
    UintahGui.setConstraints(gbc, 0, 1);
    gb.setConstraints(shearLabel, gbc);
    add(shearLabel);

    shearEntry = new DecimalField(46.0e9, 9, true);
    UintahGui.setConstraints(gbc, 1, 1);
    gb.setConstraints(shearEntry, gbc);
    add(shearEntry);

    JLabel cteLabel = new JLabel("Coeff. of Thermal Expansion");
    UintahGui.setConstraints(gbc, 0, 2);
    gb.setConstraints(cteLabel, gbc);
    add(cteLabel);

    cteEntry = new DecimalField(1.0e-5, 9, true);
    UintahGui.setConstraints(gbc, 1, 2);
    gb.setConstraints(cteEntry, gbc);
    add(cteEntry);
  }

  //--------------------------------------------------------------------
  /** Write the contents out in Uintah format */
  //--------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
      
    if (pw == null) return;

    // Write the data
    pw.println(tab+"<bulk_modulus> "+bulkEntry.getValue()+
               " </bulk_modulus>");
    pw.println(tab+"<shear_modulus> "+shearEntry.getValue()+
               " </shear_modulus>");
    pw.println(tab+"<coeff_thermal_expansion> "+cteEntry.getValue()+
               " </coeff_thermal_expansion>");

  }
}
