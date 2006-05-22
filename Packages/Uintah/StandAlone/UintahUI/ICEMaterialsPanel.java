/**************************************************************************
// Program : ICEMaterialsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           ICE materials
// Author  : Biswajit Banerjee
// Date    : 05/12/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
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
// Class   : ICEMaterialsPanel
//**************************************************************************
public class ICEMaterialsPanel extends JPanel {

  // Static variables

  // Data
  private int d_numMat = 0;
  private Vector d_geomObj = null;
  private Vector d_iceMat = null;
  private UintahInputPanel d_parent = null;

  // Local components
  private IntegerField numMatEntry = null;
  private JTabbedPane iceMatTabbedPane = null;
  private Vector iceMaterialInputPanel = null;
  
  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public ICEMaterialsPanel(Vector geomObj,
                           Vector iceMat,  
                           UintahInputPanel parent) {

    // Initialize local variables
    d_numMat = 1;
    d_geomObj = geomObj;
    d_iceMat = iceMat;
    d_parent = parent;

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Label and entry for number of materials
    JPanel numMatPanel = new JPanel(new GridLayout(1,0));
    JLabel numMatLabel = new JLabel("Number of ICE Materials");
    numMatEntry = new IntegerField(d_numMat, 2);
    numMatPanel.add(numMatLabel);
    numMatPanel.add(numMatEntry);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(numMatLabel, gbc);
    add(numMatPanel);
    
    // Create a text field listener
    TextFieldListener textListener = new TextFieldListener();
    numMatEntry.getDocument().addDocumentListener(textListener);
    
    // Create the tabbed pane
    iceMatTabbedPane = new JTabbedPane();

    // Create the panels for each material
    iceMaterialInputPanel = new Vector();
    for (int ii = 0; ii < d_numMat; ++ii) {
      ICEMaterialInputPanel matPanel = new ICEMaterialInputPanel(ii);
      iceMaterialInputPanel.addElement(matPanel);
      String matID = new String("Material "+String.valueOf(ii));
      iceMatTabbedPane.addTab(matID, null, matPanel, null);
    }
    iceMatTabbedPane.setSelectedIndex(0);
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(iceMatTabbedPane, gbc);
    add(iceMatTabbedPane);

    // Create the save button
    saveButton = new JButton("Save");
    saveButton.setActionCommand("save");
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 2, 1, 1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    saveButton.addActionListener(buttonListener);
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to changed text 
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class TextFieldListener implements DocumentListener {
    public void insertUpdate(DocumentEvent e) {
      int new_numMat = numMatEntry.getValue();
      if (new_numMat == 0) return;
      if (new_numMat < d_numMat) {
        for (int ii = d_numMat-1; ii >= new_numMat; --ii) {
          iceMaterialInputPanel.removeElementAt(ii);
          iceMatTabbedPane.remove(ii);
        }
        d_numMat = new_numMat;
      } else {
        for (int ii = d_numMat; ii < new_numMat; ++ii) {
          ICEMaterialInputPanel matPanel = new ICEMaterialInputPanel(ii);
          iceMaterialInputPanel.addElement(matPanel);
          String matID = new String("Material "+String.valueOf(ii));
          iceMatTabbedPane.addTab(matID, null, matPanel, null);
        }
        d_numMat = new_numMat;
      }
    }
    public void removeUpdate(DocumentEvent e) {
    }
    public void changedUpdate(DocumentEvent e) {
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "save") {
        
        // Create filewriter and printwriter
        File outputFile = new File("test.ups");
        try {
          FileWriter fw = new FileWriter(outputFile);
          PrintWriter pw = new PrintWriter(fw);

          String tab = new String("    ");
          for (int ii = 0; ii < d_numMat; ++ii) {
            ICEMaterialInputPanel matPanel = 
              (ICEMaterialInputPanel) iceMaterialInputPanel.elementAt(ii);
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

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ICE material input panel
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ICEMaterialInputPanel extends JPanel 
                                      implements ItemListener {

    // Data and components
    private String d_eosModel = null;
    private String d_burnModel = null;

    private JTextField matNameEntry = null;
    private JComboBox burnModelComB = null;
    private JComboBox eosModelComB = null;

    private DecimalField dynamicViscEntry = null;
    private DecimalField thermalCondEntry = null;
    private DecimalField spHeatEntry = null;
    private DecimalField speedSoundEntry = null;
    private DecimalField gammaEntry = null;

    public ICEMaterialInputPanel(int matIndex) {

      // Initialize
      d_eosModel = new String("ideal_gas");
      d_burnModel = new String("null");

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

      // Create and add the listeners
      eosModelComB.addItemListener(this);
      burnModelComB.addItemListener(this);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Listens for item picked in combo box and takes action as required.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
      String tab2 = new String(tab1+"  ");
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
      pw.println(tab+"</material>");

    }
  }
}
