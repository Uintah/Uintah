//**************************************************************************
// Class   : IceMaterialInputPanel.java
// Purpose : ICE material input panel
// Author  : Biswajit Banerjee
// Date    : 04/05/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import java.util.Vector;

public class ICEMaterialInputPanel extends JPanel 
                                   implements ItemListener,
                                              ActionListener {

  // Data and components
  private String d_eosModel = null;
  private String d_burnModel = null;
  private Vector d_geomObj = null;
  private int[] d_selGeomObj = null;

  private JTextField matNameEntry = null;
  private JComboBox burnModelComB = null;
  private JComboBox eosModelComB = null;

  private DecimalField dynamicViscEntry = null;
  private DecimalField thermalCondEntry = null;
  private DecimalField spHeatEntry = null;
  private DecimalField speedSoundEntry = null;
  private DecimalField gammaEntry = null;

  private JList geomObjectList = null;
  private DefaultListModel geomObjectListModel = null;
  private JScrollPane geomObjectSP = null;

  private JButton updateButton = null;

  public ICEMaterialInputPanel(int matIndex,
                               Vector geomObj) {

    // Initialize
    d_eosModel = new String("ideal_gas");
    d_burnModel = new String("null");
    d_geomObj = geomObj;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Add the first panel
    JPanel panel1 = new JPanel(new GridLayout(6,0));

    JLabel matNameLabel = new JLabel("Material Name");
    String matName = new String("Material "+ String.valueOf(matIndex));
    matNameEntry = new JTextField(matName, 20);
    panel1.add(matNameLabel);
    panel1.add(matNameEntry);

    JLabel dynamicViscLabel = new JLabel("Dynamic Viscosity");
    dynamicViscEntry = new DecimalField(0.0, 7);
    panel1.add(dynamicViscLabel);
    panel1.add(dynamicViscEntry);

    JLabel thermalCondLabel = new JLabel("Thermal Conductivity");
    thermalCondEntry = new DecimalField(0.0, 7);
    panel1.add(thermalCondLabel);
    panel1.add(thermalCondEntry);

    JLabel spHeatLabel = new JLabel("Specific Heat");
    spHeatEntry = new DecimalField(716.0, 7);
    panel1.add(spHeatLabel);
    panel1.add(spHeatEntry);

    JLabel speedSoundLabel = new JLabel("Speed of Sound");
    speedSoundEntry = new DecimalField(376, 7);
    panel1.add(speedSoundLabel);
    panel1.add(speedSoundEntry);

    JLabel gammaLabel = new JLabel("Gamma");
    gammaEntry = new DecimalField(1.4, 7);
    panel1.add(gammaLabel);
    panel1.add(gammaEntry);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1); 

    // Add the second panel
    JPanel panel2 = new JPanel(new GridLayout(2,0));

    JLabel eosModelLabel = new JLabel("Equation of State Model");
    eosModelComB = new JComboBox();
    eosModelComB.addItem("Ideal Gas EOS");
    eosModelComB.addItem("Murnaghan EOS");
    eosModelComB.addItem("Tillotson EOS");
    eosModelComB.addItem("Gruneisen EOS");
    eosModelComB.addItem("JWL EOS");
    eosModelComB.addItem("JWL++ EOS");
    panel2.add(eosModelLabel); panel2.add(eosModelComB); 

    JLabel burnModelLabel = new JLabel("Burn Model");
    burnModelComB = new JComboBox();
    burnModelComB.addItem("None");
    burnModelComB.addItem("Simple Burn");
    burnModelComB.addItem("Pressure Burn");
    burnModelComB.addItem("Ignition and Combustion");
    panel2.add(burnModelLabel); panel2.add(burnModelComB); 

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(panel2, gbc);
    add(panel2); 

    // Add a panel for the geometry objects
    JPanel geomObjectPanel = new JPanel(new GridLayout(1,0));
    JLabel geomObjectLabel = new JLabel("Geometry Objects");
    geomObjectPanel.add(geomObjectLabel);

    geomObjectListModel = new DefaultListModel();
    for (int ii = 0; ii < d_geomObj.size(); ++ii) {
      GeomObject go = (GeomObject) d_geomObj.elementAt(ii);
      geomObjectListModel.addElement(go.getName());
    }
    geomObjectList = new JList(geomObjectListModel);
    geomObjectList.setVisibleRowCount(4);
    geomObjectSP = new JScrollPane(geomObjectList);
    geomObjectPanel.add(geomObjectSP);

    UintahGui.setConstraints(gbc, 0, 2);
    gb.setConstraints(geomObjectPanel, gbc);
    add(geomObjectPanel);

    // Add the update button
    updateButton = new JButton("Update");
    UintahGui.setConstraints(gbc, 0, 3);
    gb.setConstraints(updateButton, gbc);
    add(updateButton);

    // Create and add the listeners
    eosModelComB.addItemListener(this);
    burnModelComB.addItemListener(this);
    updateButton.addActionListener(this);
  }

  //--------------------------------------------------------------------
  // Update action
  //--------------------------------------------------------------------
  public void actionPerformed(ActionEvent e) {
    d_selGeomObj = geomObjectList.getSelectedIndices();
    geomObjectList.setSelectedIndices(d_selGeomObj);
  }

  //--------------------------------------------------------------------
  // Listens for item picked in combo box and takes action as required.
  //--------------------------------------------------------------------
  public void itemStateChanged(ItemEvent e) {
        
    // Get the object that has been selected
    Object source = e.getItemSelectable();

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());

    if (source == burnModelComB) {
      if (item.equals(new String("None"))) {
        d_burnModel = "null";
      } else if (item.equals(new String("Simple Burn"))) {
        d_burnModel = "simple";
      } else if (item.equals(new String("Pressure Burn"))) {
        d_burnModel = "pressure";
      } else if (item.equals(new String("Ignition and Combustion"))) {
        d_burnModel = "IgnitionCombustion";
      }
    } else {
      if (item.equals(new String("Ideal Gas EOS"))) {
        d_eosModel = "ideal_gas";
      } else if (item.equals(new String("Murnaghan EOS"))) {
        d_eosModel = "Murnahan";
      } else if (item.equals(new String("Tillotson EOS"))) {
        d_eosModel = "Tillotson";
      } else if (item.equals(new String("Gruneisen EOS"))) {
        d_eosModel = "Gruneisen";
      } else if (item.equals(new String("JWL EOS"))) {
        d_eosModel = "JWL";
      } else if (item.equals(new String("JWL++ EOS"))) {
        d_eosModel = "JWLC";
      }
    }
  }

  //--------------------------------------------------------------------
  /** Write the contents out in Uintah format */
  //--------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
      
    if (pw == null) return;

    // Write the data
    String tab1 = new String(tab+"  ");
    pw.println(tab);
    pw.println(tab+"<material name = \""+matNameEntry.getText()+"\">");
    pw.println(tab1);
    pw.println(tab1+"<dynamic_viscosity> "+ dynamicViscEntry.getValue()+
               " </dynamic_viscosity>");
    pw.println(tab1+"<thermal_conductivity> "+thermalCondEntry.getValue()+
               " </thermal_conductivity>");
    pw.println(tab1+"<specific_heat> "+spHeatEntry.getValue()+
               " </specific_heat>");
    pw.println(tab1+"<speed_of_sound> "+ speedSoundEntry.getValue()+
               " </speed_of_sound>");
    pw.println(tab1+"<gamma> "+ gammaEntry.getValue()+
               " </gamma>");

    pw.println(tab1);
    pw.println(tab1+"<burn type=\""+d_burnModel+"\"/>");
    pw.println(tab1+"<EOS type=\""+d_eosModel+"\">");
    pw.println(tab1+"</EOS>");

    if (d_geomObj != null) {
      int[] numGeomObj = geomObjectList.getSelectedIndices();
      for (int ii = 0; ii < numGeomObj.length; ++ii) {
        int index = numGeomObj[ii];
        GeomObject geomObject = (GeomObject) d_geomObj.elementAt(index);        
        geomObject.writeUintah(pw, tab1);
      }
    }

    pw.println(tab+"</material>");

  }

  //-----------------------------------------------------------------------
  // Gets the name of the material
  //-----------------------------------------------------------------------
  public String getMatName() {
    return matNameEntry.getText();
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
    geomObjectListModel.removeAllElements();
    for (int ii = 0; ii < d_geomObj.size(); ++ii) {
      GeomObject go = (GeomObject) d_geomObj.elementAt(ii);
      geomObjectListModel.addElement(go.getName());
    }
    if (d_selGeomObj != null) {
      geomObjectList.setSelectedIndices(d_selGeomObj);
    }
    validate();
  }

}
