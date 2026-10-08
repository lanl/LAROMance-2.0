#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. 

==============================================================================================================

@author: Andre Ruybalid
andreruybalid@gmail.com
-----------------------

GUI for surrogate model training

"""

import sys
import os
import numpy as np
import copy
import pickle
import json
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QObject, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


from PySide6.QtWidgets import QInputDialog
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.tri as tri

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from utils import sm_build as smb
from utils import sm_data as smd
from utils import fe_shapes as fes
from utils import sm_plot as smp
from utils import sm_analyze as sma

class SurrogateModelApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        smp.darktheme(True)
        self.setWindowTitle("Surrogate Model Builder")
        self.setGeometry(100, 100, 1600, 900)

        # Apply modern dark theme stylesheet
        self._apply_modern_dark_theme()

        self.state_file = "references/last_gui_state.json"
        self.editable_labels = {}  # Store for editable labels


        self.data = None
        self.input_keys = []
        self.output_keys = []
        self.roi = None
        self.mesh_specs = {}
        self.highlight_elements = []
        self.args = {}
        
        self.rectangle_selector = None

        self.preview_mesh = False
        self.enable_lasso_selector = False

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.pca_canvas = None  
        self.ts_canvas = None  


        self.setup_panel = QtWidgets.QWidget()
        self.setup_layout = QtWidgets.QVBoxLayout(self.setup_panel)

        self.tabs = QtWidgets.QTabWidget()
        self.data_tab = QtWidgets.QWidget()
        self.result_tab = QtWidgets.QWidget()
        self.dict_tab = QtWidgets.QWidget()

        self.tabs.addTab(self.data_tab, "Data Preview")
        self.tabs.addTab(self.result_tab, "Training Result")
        self.tabs.addTab(self.dict_tab, "Review Configs")

        self.data_tab_layout = QtWidgets.QVBoxLayout(self.data_tab)
        self.result_tab_layout = QtWidgets.QVBoxLayout(self.result_tab)
        self.dict_tab_layout = QtWidgets.QVBoxLayout(self.dict_tab)

        self.result_options_layout = QtWidgets.QHBoxLayout()
    

        # Create stacked data preview layout to hold multiple pages (view)
        self.data_plot_stack = QtWidgets.QStackedLayout()
        self.data_tab_layout.addLayout(self.data_plot_stack)
        self.data_canvas_2d = None
        self.data_canvas_3d = None


        # Create stacked results layout to hold multiple pages (views)
        self.result_plot_stack = QtWidgets.QStackedWidget()
        
        self.result_tab_layout.addWidget(self.result_plot_stack)

        btn_row = QtWidgets.QHBoxLayout()
        self.prev_plot_btn = QtWidgets.QPushButton("← 2D View")
        self.next_plot_btn = QtWidgets.QPushButton("3D View →")

        self.pca_button = QtWidgets.QPushButton("Nodal PCA")
        self.pca_button.clicked.connect(self.show_pca_heatmap_canvas)

        self.plot_ts_button = QtWidgets.QPushButton("Plot Time Series\n(beta)")
        self.plot_ts_button.setToolTip("Plot stacked time-series for selected output")
        self.plot_ts_button.clicked.connect(self.on_time_series_plot_button_clicked)
        
         # or add to appropriate layout
        btn_row.addWidget(self.prev_plot_btn)
        btn_row.addWidget(self.next_plot_btn)
        btn_row.addWidget(self.pca_button) 
        btn_row.addWidget(self.plot_ts_button)
        
        # Insert Load Training button before other buttons in the result tab
        load_training_btn = QtWidgets.QPushButton("Load Training")
        load_training_btn.clicked.connect(self.load_training_result)
        btn_row.insertWidget(0, load_training_btn)

        self.result_tab_layout.insertLayout(0, btn_row)

        self.prev_plot_btn.clicked.connect(self.show_prev_plot)
        self.next_plot_btn.clicked.connect(self.show_next_plot)

        self.toggle_data_points = QtWidgets.QCheckBox("Show Data Points")
        self.toggle_data_points.setChecked(True)
        self.toggle_data_points.stateChanged.connect(self.update_result_plot)

        self.toggle_trisurf = QtWidgets.QCheckBox("Show TriSurf")
        self.toggle_trisurf.setChecked(True)
        self.toggle_trisurf.stateChanged.connect(self.update_result_plot)

        self.toggle_mapped_output_in_plots = QtWidgets.QCheckBox("Transform Outputs")
        self.toggle_mapped_output_in_plots.setChecked(True)
        self.toggle_mapped_output_in_plots.stateChanged.connect(self.update_result_plot)

        self.toggle_nodes = QtWidgets.QCheckBox("Show Nodes")
        self.toggle_nodes.setChecked(True)
        self.toggle_nodes.stateChanged.connect(self.update_result_plot)

        self.result_options_layout.addWidget(self.toggle_data_points)
        self.result_options_layout.addWidget(self.toggle_trisurf)
        self.result_options_layout.addWidget(self.toggle_mapped_output_in_plots)
        self.result_options_layout.addWidget(self.toggle_nodes)
        self.result_tab_layout.addLayout(self.result_options_layout)

        # --- Colormap Selector ---
        self.colormap_selector = QtWidgets.QComboBox()
        self.colormap_selector.addItems(["hot", "viridis", "plasma", "inferno", "cividis", "magma", "coolwarm"])
        self.colormap_selector.setCurrentText("hot")
        self.colormap_selector.currentTextChanged.connect(self.update_result_plot)
        self.current_cmap = "hot"

        label = QtWidgets.QLabel("Colormap:")
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(1)
        row.setContentsMargins(0, 0, 0, 0)  # Optional: remove internal margins too
        row.addWidget(label)
        row.addWidget(self.colormap_selector)

        row_widget = QtWidgets.QWidget()
        row_widget.setLayout(row)

        self.result_options_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        self.result_options_layout.addWidget(self.colormap_selector)
        
        # Set up figure export controls
        self.setup_result_export_controls()

    
        self.dict_text = QtWidgets.QTextEdit()
        self.dict_tab_layout.addWidget(self.dict_text)

        self.main_layout.addWidget(self.setup_panel)
        self.main_layout.addWidget(self.tabs)

        self.main_layout.setStretch(0, 4)
        self.main_layout.setStretch(1, 6)

        # Each input widget: (combo box, elem_input, tri_checkbox, plot_checkbox)
        self.input_var_widgets = []
        self.output_var_cbs = []

        self.plot_2d_mode = False  # Default 3D plot

        self.setup_ui()

        self.data_canvas = None
        self.result_canvas = None
        
        self.load_state()  

    def _apply_modern_dark_theme(self):
        """Black-metallic dark theme with silver accents and subtle gradients."""
        qss = """
        /* === Global === */
        QMainWindow {
            background-color: #0f0f0f;
        }
        QWidget {
            background-color: #0f0f0f;
            color: #d4d4d4;
            font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
        }

        /* === Group Boxes (Cards) === */
        QGroupBox {
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px;
            background-color: #1a1a1c;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 10px;
            background-color: #1a1a1c;
            color: #c0c0c0;
            font-weight: 600;
            font-size: 13px;
        }

        /* === Buttons (metallic look) === */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2a2a2c, stop:1 #1f1f21);
            color: #e0e0e0;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 6px 14px;
            min-height: 28px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3a3a3c, stop:1 #2a2a2c);
            border-color: #a8a8a8;
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a1a1c, stop:1 #121212);
        }

        /* Primary action buttons */
        QPushButton[text*="Run Training"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #22c55e, stop:1 #16a34a);
            color: white;
            font-weight: 600;
            border: none;
        }
        QPushButton[text*="Run Training"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #16a34a, stop:1 #15803d);
        }

        QPushButton[text*="Load Training"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #a8a8a8, stop:1 #7a7a7a);
            color: black;
            font-weight: 600;
            border: none;
        }
        QPushButton[text*="Load Training"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #c0c0c0, stop:1 #8a8a8a);
        }

        QPushButton[text*="Preview Mesh"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #a8a8a8, stop:1 #7a7a7a);
            color: white;
            font-weight: 500;
            border: none;
        }
        QPushButton[text*="Preview Mesh"]:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #c0c0c0, stop:1 #8a8a8a);
        }

        /* Small +/- buttons */
        QPushButton[text="➕"], QPushButton[text="➖"] {
            background: #1f1f21;
            border: 1px solid #3a3a3a;
            padding: 2px;
            font-size: 14px;
            min-width: 22px;
            max-width: 22px;
            min-height: 22px;
            max-height: 22px;
        }
        QPushButton[text="➕"]:hover { background: #22c55e; color: white; }
        QPushButton[text="➖"]:hover { background: #ef4444; color: white; }

        /* === Combo Boxes & Line Edits === */
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #121212;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
            padding: 4px 8px;
            color: #d4d4d4;
        }
        QComboBox:hover, QLineEdit:hover {
            border-color: #a8a8a8;
        }
        QComboBox::drop-down {
            border: none;
        }

        /* === Checkboxes === */
        QCheckBox {
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #a8a8a8;
            border-radius: 3px;
            background-color: #121212;
        }
        QCheckBox::indicator:checked {
            background-color: #a8a8a8;
            image: url(none);
        }

        /* === Tabs === */
        QTabWidget::pane {
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            background: #1a1a1c;
        }
        QTabBar::tab {
            background: #121212;
            color: #a1a1aa;
            padding: 8px 18px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #a8a8a8, stop:1 #7a7a7a);
            color: white;
            font-weight: 600;
        }
        QTabBar::tab:hover {
            background: #2a2a2c;
            color: #e0e0e0;
        }

        /* === Text Edit (Config Preview) === */
        QTextEdit {
            background-color: #121212;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 12px;
        }

        /* === Scrollbars === */
        QScrollBar:vertical {
            background: #121212;
            width: 8px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #a8a8a8, stop:1 #7a7a7a);
            min-height: 20px;
            border-radius: 4px;
        }
        """
        self.setStyleSheet(qss)

    def show_prev_plot(self):
        if self.data_canvas_2d is not None:
            self.result_plot_stack.setCurrentWidget(self.data_canvas_2d)
    def show_next_plot(self):
        if self.data_canvas_3d is not None:
            self.result_plot_stack.setCurrentWidget(self.data_canvas_3d)

    def _hbox(self, *widgets):
        box = QtWidgets.QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QtWidgets.QWidget()
        container.setLayout(box)
        return container

    def get_refinement_settings(self):
        if not self.enable_checkbox.isChecked():
            return None, None
        var1 = self.var1_cb.currentText()
        var2 = self.var2_cb.currentText()

        region = {
            var1: (float(self.min1.text()), float(self.max1.text())),
            var2: (float(self.min2.text()), float(self.max2.text()))
        }
        configs = {
            "spacing": float(self.spacing_box.value()),
            "mode": self.mode_box.currentText()
        }
        return region, configs

    def remove_highlighted_elements(self):
        if not hasattr(self, "highlight_elements") or not self.highlight_elements:
            QtWidgets.QMessageBox.information(self, "No Elements", "No highlighted elements to remove.")
            return

        nodes = self.preview_nodes
        conn = self.preview_conn  # make sure you store this when mesh is built
        elems_to_remove = self.highlight_elements

        # Remove the elements
        new_nodes, new_conn = fes.remove_elements_by_index(nodes, conn, elems_to_remove, enforce_diagonals=False)
        new_conn = fes.enforce_checkerboard_diagonals(new_conn, new_nodes)

        # Update memory
        self.preview_nodes = new_nodes
        self.preview_conn = new_conn
        self.mesh_specs['premade_mesh']['nodes'] = new_nodes.tolist()
        self.mesh_specs['premade_mesh']['conn'] = new_conn.tolist()
        self.highlight_elements = []
        self.mesh_specs['mesh_filtering']['highlight_elements'] = []

        # Redraw mesh
        self.plot_updated_mesh()

    def plot_updated_mesh(self):
        if self.data_canvas_2d is None:
            print("No canvas found to update.")
            return

        fig = self.data_canvas_2d.figure
        if not fig.axes:
            print("No axes in figure.")
            return

        ax2d = fig.axes[0]
        ax2d.clear()

        # Redraw data scatter points
        selected_inputs = [w for w in self.input_var_widgets if w[3].isChecked()]
        var1 = selected_inputs[0][0].currentText()
        var2 = selected_inputs[1][0].currentText()
        selected_output = None
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                selected_output = cb.currentText()
                break

        x = self.data['data'][var1]
        y = self.data['data'][var2]
        z = self.data['data']['U'][selected_output]

        max_points = self.preview_points_spinbox.value()
        indices = np.random.choice(len(x), min(max_points, len(x)), replace=False)

        x = x[indices]
        y = y[indices]
        z = z[indices]

        if np.min(z) < 0.0:
            print("Updated mesh with symlog-colors")
            norm = mcolors.SymLogNorm(vmin=np.min(z), vmax=np.max(z), linthresh=0.01)
        else:
            print("Updated mesh with logarithmic colors")
            norm = mcolors.LogNorm(vmin=np.min(z), vmax=np.max(z))

        ax2d.scatter(x, y, c=z, cmap='viridis', norm=norm, s=10)

        # Redraw updated mesh (no highlights)
        smp.visualize_mesh(
            self.preview_nodes,
            self.preview_conn,
            dim=2,
            show_elem_numbers=False,
            show_node_numbers=False,
            ax=ax2d
        )

        self.data_canvas_2d.draw()


    def setup_result_export_controls(self):
        # Create a save menu button with dropdown
        self.save_menu = QtWidgets.QMenu()
        
        # Add actions for different save options
        self.save_current_action = self.save_menu.addAction("Save Current Figure")
        self.save_current_action.triggered.connect(self.save_current_figure)
        
        self.save_all_action = self.save_menu.addAction("Save All Figures")
        self.save_all_action.triggered.connect(self.save_all_figures)
        
        self.export_data_action = self.save_menu.addAction("Export Data as CSV")
        self.export_data_action.triggered.connect(self.export_figure_data)
        
        # Create the main button and set the menu
        self.save_figure_btn = QtWidgets.QPushButton("Export")
        self.save_figure_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_figure_btn.setMenu(self.save_menu)
        
        # Add to existing layout
        self.result_options_layout.addWidget(self.save_figure_btn)
        
        # Add theme toggle
        self.theme_toggle = QtWidgets.QCheckBox("Dark Theme")
        self.theme_toggle.setChecked(True)  # Default is dark theme
        self.theme_toggle.setToolTip("Toggle between dark and light theme for plots")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        self.result_options_layout.addWidget(self.theme_toggle)
        
    def save_current_figure(self):
        """Save the currently displayed figure to file with advanced options"""
        # Get the current widget
        current_widget = self.result_plot_stack.currentWidget()
        
        if current_widget is None:
            QtWidgets.QMessageBox.warning(self, "No Figure", "No figure available to save.")
            return
        
        # Special handling for time series widget which has a nested canvas
        figure = None
        if hasattr(self, 'ts_widget') and current_widget == self.ts_widget and hasattr(self, 'ts_canvas'):
            # Time series case - get figure from the canvas
            figure = self.ts_canvas.figure
        else:
            # Regular case - direct figure access
            if hasattr(current_widget, 'figure'):
                figure = current_widget.figure
            else:
                QtWidgets.QMessageBox.warning(self, "No Figure", 
                    "Cannot access figure from current view.")
                return
        
        # Determine type of figure for default filename suggestion
        figure_type = "unknown"
        if current_widget == self.data_canvas_2d:
            figure_type = "Training_2D_plot"
        elif current_widget == self.data_canvas_3d:
            figure_type = "Training_3D_plot"
        elif current_widget == self.pca_canvas:
            figure_type = "Training_PCA_heatmap"
        elif current_widget == self.ts_widget:
            figure_type = "Training_time_series"
        
        # Get the selected output variable for filename
        output_var = "output"
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                output_var = cb.currentText()
                break
        
        # Create suggested filename
        suggested_name = f"{figure_type}_{output_var}"
        
        # Show advanced save dialog with the retrieved figure
        self._show_figure_save_dialog(figure, suggested_name)

    def _show_figure_save_dialog(self, figure, suggested_name):
        """Show advanced save dialog with options for the figure"""
        # Create custom dialog for advanced options
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save Figure Options")
        layout = QtWidgets.QVBoxLayout()
        
        # Get original title
        original_title = ""
        if figure.axes:
            # Try to get title from first axis
            original_title = figure.axes[0].get_title()
            # If no axis title, try figure suptitle
            if not original_title and figure._suptitle and figure._suptitle.get_text():
                original_title = figure._suptitle.get_text()
        
        # Title input field
        title_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("Figure Title:")
        title_edit = QtWidgets.QLineEdit(original_title)
        title_layout.addWidget(title_label)
        title_layout.addWidget(title_edit)

        # File path selection
        file_layout = QtWidgets.QHBoxLayout()
        file_label = QtWidgets.QLabel("Save to:")
        file_path = QtWidgets.QLineEdit(f"{suggested_name}.png")
        browse_btn = QtWidgets.QPushButton("Browse...")
        
        def browse_file():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog, "Save Figure", file_path.text(),
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;EPS Files (*.eps)"
            )
            if path:
                file_path.setText(path)
        
        browse_btn.clicked.connect(browse_file)
        file_layout.addWidget(file_label)
        file_layout.addWidget(file_path)
        file_layout.addWidget(browse_btn)
        
        # DPI setting
        dpi_layout = QtWidgets.QHBoxLayout()
        dpi_label = QtWidgets.QLabel("Resolution (DPI):")
        dpi_spinner = QtWidgets.QSpinBox()
        dpi_spinner.setRange(72, 1200)
        dpi_spinner.setValue(300)
        dpi_spinner.setSingleStep(50)
        dpi_layout.addWidget(dpi_label)
        dpi_layout.addWidget(dpi_spinner)
        
        # Figure size
        size_layout = QtWidgets.QHBoxLayout()
        size_label = QtWidgets.QLabel("Size adjustment:")
        size_spinner = QtWidgets.QDoubleSpinBox()
        size_spinner.setRange(0.5, 5.0)
        size_spinner.setValue(1.0)
        size_spinner.setSingleStep(0.1)
        size_spinner.setPrefix("× ")
        size_layout.addWidget(size_label)
        size_layout.addWidget(size_spinner)
        
        # Transparent background option
        transparent_cb = QtWidgets.QCheckBox("Transparent background")
        
        # Theme selection
        theme_label = QtWidgets.QLabel("Background Theme:")
        theme_combo = QtWidgets.QComboBox()
        theme_combo.addItems(["Current", "Dark", "Light"])
        theme_combo.setCurrentText("Current")
        theme_combo.setToolTip("Override current theme when saving")
        
        theme_layout = QtWidgets.QHBoxLayout()
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(theme_combo)
        
        # Axis labels section
        axis_group = QtWidgets.QGroupBox("Customize Axis Labels")
        axis_layout = QtWidgets.QFormLayout()
        
        # Get current labels from the figure
        x_label = figure.axes[0].get_xlabel() if figure.axes and hasattr(figure.axes[0], 'get_xlabel') else ""
        y_label = figure.axes[0].get_ylabel() if figure.axes and hasattr(figure.axes[0], 'get_ylabel') else ""
        z_label = ""
        
        # Check if this is a 3D plot with z-axis
        for ax in figure.axes:
            if hasattr(ax, 'get_zlabel'):
                z_label = ax.get_zlabel()
                break
        
        x_label_edit = QtWidgets.QLineEdit(x_label)
        y_label_edit = QtWidgets.QLineEdit(y_label)
        z_label_edit = QtWidgets.QLineEdit(z_label)
        
        axis_layout.addRow("X-axis label:", x_label_edit)
        axis_layout.addRow("Y-axis label:", y_label_edit)
        
        # Only show Z-label field for 3D plots
        z_label_row = None
        if any(hasattr(ax, 'get_zlabel') for ax in figure.axes):
            axis_layout.addRow("Z-axis label:", z_label_edit)
        else:
            z_label_edit.setVisible(False)
            
        axis_group.setLayout(axis_layout)
        
        # Add all to main layout
        layout.addLayout(title_layout)
        layout.addLayout(file_layout)
        layout.addLayout(dpi_layout)
        layout.addLayout(size_layout)
        layout.addWidget(transparent_cb)
        layout.addLayout(theme_layout)
        layout.addWidget(axis_group)
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        # Execute dialog
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            try:
                # Get values
                save_path = file_path.text()
                dpi = dpi_spinner.value()
                size_factor = size_spinner.value()
                transparent = transparent_cb.isChecked()
                
                # Get theme selection
                theme_selection = theme_combo.currentText()
                is_dark_theme = self.theme_toggle.isChecked()  # Current theme state
                
                # Store original properties to restore later
                orig_size = figure.get_size_inches()
                
                # Store original title
                original_title = ""
                has_suptitle = False
                if figure._suptitle and figure._suptitle.get_text():
                    original_title = figure._suptitle.get_text()
                    has_suptitle = True
                
                # Store original axis labels
                original_labels = {}
                for ax in figure.axes:
                    original_labels[ax] = {
                        'xlabel': ax.get_xlabel() if hasattr(ax, 'get_xlabel') else None,
                        'ylabel': ax.get_ylabel() if hasattr(ax, 'get_ylabel') else None,
                        'zlabel': ax.get_zlabel() if hasattr(ax, 'get_zlabel') else None,
                        'title': ax.get_title() if hasattr(ax, 'get_title') else None
                    }
                
                # Apply custom axis labels - only to main plot axes, not colorbars
                main_axes = []
                for i, ax in enumerate(figure.axes):
                    # Simple heuristic to identify main plot axes vs. colorbars
                    # Main axes typically have a larger area than colorbar axes
                    bbox = ax.get_position()
                    width = bbox.width
                    height = bbox.height
                    area = width * height
                    
                    # Identify main axes (usually the first axis or ones with larger area)
                    if i == 0 or area > 0.2:  # Main plot typically has an area > 0.2
                        main_axes.append(ax)

                # Apply labels only to main plot axes
                for ax in main_axes:
                    if hasattr(ax, 'set_xlabel'):
                        ax.set_xlabel(x_label_edit.text())
                    if hasattr(ax, 'set_ylabel'):
                        ax.set_ylabel(y_label_edit.text())
                    if hasattr(ax, 'set_zlabel') and z_label_edit.isVisible():
                        ax.set_zlabel(z_label_edit.text())
                
                # Clear existing titles to prevent overlapping
                # First store original axis titles (already done in original_labels)
                # Clear figure suptitle
                if figure._suptitle:
                    figure._suptitle.set_text("")
                
                # Clear all axis titles temporarily
                for ax in figure.axes:
                    if hasattr(ax, 'set_title'):
                        ax.set_title("")
                
                # Apply custom title as figure suptitle
                new_title = title_edit.text()
                if new_title:
                    figure.suptitle(new_title)
                
                # Apply custom theme if selected
                if theme_selection != "Current":
                    applied_theme = theme_selection == "Dark"
                    self._apply_theme_to_figure(figure, applied_theme)
                
                # Adjust figure size if needed
                figure.set_size_inches(orig_size[0] * size_factor, orig_size[1] * size_factor)
                
                # Save with options
                figure.savefig(
                    save_path,
                    dpi=dpi,
                    transparent=transparent,
                    bbox_inches='tight'
                )
                
                # Restore original size, labels and theme
                figure.set_size_inches(orig_size[0], orig_size[1])
                
                # Restore original title state
                if has_suptitle:
                    figure.suptitle(original_title)
                else:
                    # Make sure to completely remove suptitle if there wasn't one originally
                    if figure._suptitle:
                        figure._suptitle.remove()
                        figure._suptitle = None
                
                # Restore original labels
                for ax, labels in original_labels.items():
                    if labels['xlabel'] is not None:
                        ax.set_xlabel(labels['xlabel'])
                    if labels['ylabel'] is not None:
                        ax.set_ylabel(labels['ylabel'])
                    if labels['zlabel'] is not None:
                        ax.set_zlabel(labels['zlabel'])
                    if labels['title'] is not None:
                        ax.set_title(labels['title'])
                
                # Restore original theme if changed
                if theme_selection != "Current":
                    self._apply_theme_to_figure(figure, is_dark_theme)
                
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Figure saved successfully to {save_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to save figure: {str(e)}"
                )
                
    def save_all_figures(self):
        """Save all available figures to a selected directory"""
        # Create custom dialog with format options
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save Figures")
        layout = QtWidgets.QVBoxLayout()

        # Format selection
        format_layout = QtWidgets.QHBoxLayout()
        format_label = QtWidgets.QLabel("File Format:")
        format_combo = QtWidgets.QComboBox()
        format_combo.addItems(["PNG (.png)", "PDF (.pdf)", "SVG (.svg)"])
        format_layout.addWidget(format_label)
        format_layout.addWidget(format_combo)
        layout.addLayout(format_layout)

        # DPI setting (only for raster formats)
        dpi_layout = QtWidgets.QHBoxLayout()
        dpi_label = QtWidgets.QLabel("Resolution (DPI):")
        dpi_spinner = QtWidgets.QSpinBox()
        dpi_spinner.setRange(72, 1200)
        dpi_spinner.setValue(300)
        dpi_spinner.setSingleStep(50)
        dpi_layout.addWidget(dpi_label)
        dpi_layout.addWidget(dpi_spinner)
        layout.addLayout(dpi_layout)
        
        # Theme override
        theme_layout = QtWidgets.QHBoxLayout()
        theme_label = QtWidgets.QLabel("Theme:")
        theme_combo = QtWidgets.QComboBox()
        theme_combo.addItems(["Current", "Dark", "Light"])
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(theme_combo)
        layout.addLayout(theme_layout)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        
        # Update DPI visibility based on format selection
        def update_dpi_visibility(index):
            format_name = format_combo.currentText().lower()
            dpi_label.setVisible('png' in format_name)
            dpi_spinner.setVisible('png' in format_name)
            
        format_combo.currentIndexChanged.connect(update_dpi_visibility)
        
        # Show dialog
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
            
        # Get selected options
        format_text = format_combo.currentText()
        if "PNG" in format_text:
            file_ext = ".png"
        elif "PDF" in format_text:
            file_ext = ".pdf"
        elif "SVG" in format_text:
            file_ext = ".svg"
        else:
            file_ext = ".png"  # Default
            
        dpi = dpi_spinner.value()
        selected_theme = theme_combo.currentText()
        
        # Ask for directory
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory for Figures"
        )
        
        if not directory:
            return
            
        # Get the selected output variable for filename
        output_var = "output"
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                output_var = cb.currentText()
                break
                
        saved_count = 0
        error_count = 0
        
        # Try to save each available figure
        figures_to_save = []
        
        # Add available canvases and their base filenames
        if hasattr(self, 'data_canvas_2d') and self.data_canvas_2d:
            figures_to_save.append((self.data_canvas_2d.figure, "Training_2D_plot"))
            
        if hasattr(self, 'data_canvas_3d') and self.data_canvas_3d:
            figures_to_save.append((self.data_canvas_3d.figure, "Training_3D_plot"))
            
        if hasattr(self, 'pca_canvas') and self.pca_canvas:
            figures_to_save.append((self.pca_canvas.figure, "Training_PCA_heatmap"))
            
        # Special handling for time series which has a nested canvas
        if hasattr(self, 'ts_canvas') and self.ts_canvas:
            figures_to_save.append((self.ts_canvas.figure, "Training_time_series"))
        elif hasattr(self, 'ts_widget') and hasattr(self, 'ts_canvas'):
            # Access the figure from the canvas within the widget
            figures_to_save.append((self.ts_canvas.figure, "Training_time_series"))
        
        # Get current theme state
        is_dark_theme = self.theme_toggle.isChecked()
        
        # Save each figure
        for fig, base_name in figures_to_save:
            try:
                filename = f"{base_name}_{output_var}{file_ext}"
                filepath = os.path.join(directory, filename)
                
                # Store original theme to restore later
                if selected_theme != "Current":
                    # Apply selected theme temporarily
                    applied_theme = selected_theme == "Dark"
                    self._apply_theme_to_figure(fig, applied_theme)
                
                # Save with appropriate settings
                fig.savefig(
                    filepath,
                    dpi=dpi if file_ext == '.png' else None,  # DPI only for raster formats
                    bbox_inches='tight'
                )
                
                # Restore original theme if changed
                if selected_theme != "Current":
                    self._apply_theme_to_figure(fig, is_dark_theme)
                    
                saved_count += 1
            except Exception as e:
                print(f"Error saving {base_name}: {e}")
                error_count += 1
        
        # Show results
        if saved_count > 0:
            msg = f"Successfully saved {saved_count} figures to {directory}"
            if error_count > 0:
                msg += f"\n{error_count} figures could not be saved."
            QtWidgets.QMessageBox.information(self, "Figures Saved", msg)
        else:
            QtWidgets.QMessageBox.warning(
                self, "No Figures Saved", 
                f"No figures were saved. Either no figures were available or all save operations failed."
            )

    def export_figure_data(self):
        """Export the data from the current figure to a CSV file"""
        # Get the current widget
        current_widget = self.result_plot_stack.currentWidget()
        
        if current_widget is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No figure data available to export.")
            return
        
        # Ask for file location
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Get relevant data based on current view
            import pandas as pd
            
            # Get data for export
            if not hasattr(self, 'input_vars') or not self.input_vars:
                raise ValueError("No input variables available")
                
            # Get selected output variable
            output_var = None
            for cb, plot_cb, _, _ in self.output_var_cbs:
                if plot_cb.isChecked():
                    output_var = cb.currentText()
                    break
                    
            if not output_var:
                raise ValueError("No output variable selected")
            
            # Create DataFrame with input variables
            data_dict = {}
            for var in self.input_vars:
                if var in self.DATA2['data']:
                    data_dict[var] = self.DATA2['data'][var]
                    
            # Add output variables if available
            if hasattr(self, 'outputs') and output_var in self.outputs:
                output_data = self.outputs[output_var]
                data_dict[f"{output_var}_data"] = output_data["u_physical"]
                data_dict[f"{output_var}_fit"] = output_data["uh_physical"]
                
                # Add relative error
                rel_err = 100 * np.abs(
                    (output_data["u_physical"] - output_data["uh_physical"]) / 
                    np.where(output_data["u_physical"] == 0, np.finfo(float).eps, output_data["u_physical"])
                )
                data_dict[f"{output_var}_rel_error_pct"] = rel_err
                
            # Create and save DataFrame
            df = pd.DataFrame(data_dict)
            df.to_csv(filepath, index=False)
            
            QtWidgets.QMessageBox.information(
                self, "Success", f"Data exported successfully to {filepath}"
            )
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to export data: {str(e)}"
            )
    
    def open_mesh_settings_dialog(self):
        dlg = MeshSettingsDialog(self, self.input_keys)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            region, configs = dlg.get_refinement_settings()
            if region and configs:
                self.mesh_specs["refinement_region"] = region
                self.mesh_specs["refinement_configs"] = configs
            else:
                self.mesh_specs["refinement_region"] = None

    def truncate_path(self, path, max_length=100):
        if len(path) <= max_length:
            return path
        else:
            # Show start and end of path, with "..." in the middle
            part_len = (max_length - 3) // 2
            return path[:part_len] + "..." + path[-part_len:]

    # -------------------------------------------------------------------------
    # Helper to keep the file‑path label concise and multi‑line
    # -------------------------------------------------------------------------
    def _update_file_path_label(self):
        """
        Update ``self.file_path_label`` to show a concise, wrapped list of loaded files.
        Each path is truncated to a reasonable length and placed on its own line.
        Only the most recent few files are displayed to keep the GUI usable.
        """
        max_display_len = 80  # characters per line before truncation
        # Ensure we have a list to work with
        if not hasattr(self, "loaded_files") or not self.loaded_files:
            self.file_path_label.setText("No file loaded")
            return

        # Truncate each path and join with newlines
        truncated = [self.truncate_path(p, max_length=max_display_len) for p in self.loaded_files]
        display_text = "\n".join(truncated)
        self.file_path_label.setText(display_text)
    
    def _normalize_roi(self):
        """
        Ensure ROI entries are lists of two floats.
        This method normalizes ROI values that may have been stored as scalars,
        single‑element lists/tuples, or NumPy scalars into the expected
        ``[min, max]`` list format.
        """
        if not isinstance(self.roi, dict):
            return
        for key, val in list(self.roi.items()):
            # Scalar values – duplicate for min and max
            if isinstance(val, (int, float, np.integer, np.floating)):
                self.roi[key] = [float(val), float(val)]
            # List/tuple/ndarray values
            elif isinstance(val, (list, tuple, np.ndarray)):
                arr = list(val)
                if len(arr) == 1:
                    self.roi[key] = [float(arr[0]), float(arr[0])]
                elif len(arr) >= 2:
                    self.roi[key] = [float(arr[0]), float(arr[1])]
            # Anything else – leave unchanged

    def load_state(self):
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            self.roi = state.get("roi", None)
            self._normalize_roi()

            # Restore mesh specs and args early so we can extract data_file
            self.mesh_specs = state.get("mesh_specs", {})
            self.args = state.get("args", {})

            # Try to reload data file BEFORE setting up input/output widgets
            data_file = self.args.get("data_file", None)
            if data_file:
                try:
                    self.load_data_from_file(data_file, reset_ui=False)
                    print(f"Data reloaded from {data_file}")
                except Exception as data_err:
                    print(f"Warning: Failed to reload data from file {data_file}: {data_err}")
            else:
                print("No data_file path found in args.")

            # Clear existing inputs and outputs (UI widgets)
            self.reset_inputs_outputs()

            # If state has inputs/outputs saved, rebuild them; else add defaults
            inputs_saved = state.get("inputs", [])
            outputs_saved = state.get("outputs", [])

            if inputs_saved and outputs_saved:
                # Rebuild inputs from saved state
                for input_cfg in inputs_saved:
                    self.add_input_cb()
                    cb, elem_input, tri_cb, plot_cb, map_dropdown, scaler_btn = self.input_var_widgets[-1]
                    
                    # Set basic UI properties
                    key = input_cfg.get("key", "")
                    cb.setCurrentText(key)
                    elem_input.setText(input_cfg.get("elements", "6"))
                    tri_cb.setChecked(input_cfg.get("tri", False))
                    plot_cb.setChecked(input_cfg.get("plot", False))
                    
                    # Set mapping dropdown
                    mapping_type = input_cfg.get("mapping", "minmax")
                    idx = map_dropdown.findText(mapping_type)
                    if idx >= 0:
                        map_dropdown.setCurrentIndex(idx)
                    
                    # Restore transform configuration if available
                    if "transform_config" in input_cfg and key:
                        if "map_input" not in self.args:
                            self.args["map_input"] = {}
                        self.args["map_input"][key] = input_cfg["transform_config"]

                # Rebuild outputs from saved state
                for out_entry in outputs_saved:
                    self.add_output_cb()
                    cb, plot_cb, map_dropdown, bounds_cb = self.output_var_cbs[-1]
                    
                    # Set basic UI properties
                    key = out_entry.get("key", "")
                    cb.setCurrentText(key)
                    plot_cb.setChecked(out_entry.get("plot", False))
                    
                    # Set mapping dropdown
                    mapping_type = out_entry.get("map", "minmax")
                    idx = map_dropdown.findText(mapping_type)
                    if idx >= 0:
                        map_dropdown.setCurrentIndex(idx)
                    
                    # Restore bounds checkbox and values
                    bounds_checked = out_entry.get("bounds", False)
                    bounds_cb.setChecked(bounds_checked)
                    
                    # Restore transform configuration if available
                    if "transform_config" in out_entry and key:
                        if "map_output" not in self.args:
                            self.args["map_output"] = {}
                        self.args["map_output"][key] = out_entry["transform_config"]
                    
                    # Restore bounds values if available
                    if "bounds_values" in out_entry and key:
                        if "constrain_regression" not in self.args:
                            self.args["constrain_regression"] = {}
                        self.args["constrain_regression"][key] = out_entry["bounds_values"]
                        # Store bounds values directly on combobox for reference
                        cb.bounds_values = out_entry["bounds_values"]
            else:
                # No saved inputs/outputs, so add default inputs/outputs
                if self.input_keys:
                    for i in range(min(2, len(self.input_keys))):
                        self.add_input_cb()
                        cb, elem_input, tri_cb, plot_cb, map_dropdown, scaler_btn = self.input_var_widgets[-1]
                        cb.setCurrentText(self.input_keys[i])
                        elem_input.setText("6")
                        tri_cb.setChecked(False)
                        plot_cb.setChecked(False)
                        map_dropdown.setCurrentIndex(map_dropdown.findText("minmax"))
                else:
                    for _ in range(2):
                        self.add_input_cb()

                if self.output_keys:
                    self.add_output_cb()
                    cb, plot_cb, map_dropdown, bounds_cb = self.output_var_cbs[-1]  # unpack 4
                    cb.setCurrentText(self.output_keys[0])
                else:
                    self.add_output_cb()

            print("State loaded.")
        except Exception as e:
            print(f"Error loading state: {e}")

            # In case of failure, add defaults so UI is not empty
            self.reset_inputs_outputs()
            for _ in range(2):
                self.add_input_cb()
            self.add_output_cb()


    # def save_state(self):
    #     try:
    #         state = {
    #             "roi": self.roi,
    #             "plot_2d_mode": self.plot_2d_mode,
    #             "inputs": [],
    #             "outputs": [cb.currentText() for cb, _, _, _ in self.output_var_cbs],
    #             "mesh_specs": self.mesh_specs,
    #             "args": self.args,
    #         }

    #         for cb, elem_input, tri_cb, plot_cb, map_dropdown, scaler_btn in self.input_var_widgets:
    #             state["inputs"].append({
    #                 "key": cb.currentText(),
    #                 "elements": elem_input.text(),
    #                 "tri": tri_cb.isChecked(),
    #                 "plot": plot_cb.isChecked(),
    #                 "mapping": map_dropdown.currentText()
    #             })

    #         with open(self.state_file, "w") as f:
    #             json.dump(state, f, indent=2)
    #         print("State saved.")
    #     except Exception as e:
    #         print(f"Error saving state: {e}")

    # ENHANCED: Save complete transform settings
    def save_state(self):
        try:
            state = {
                "roi": self.roi,
                "plot_2d_mode": self.plot_2d_mode,
                "inputs": [],
                "outputs": [],
                "mesh_specs": self.mesh_specs,
                "args": self.args,
            }

            # Save input widgets with complete transform settings
            for cb, elem_input, tri_cb, plot_cb, map_dropdown, scaler_btn in self.input_var_widgets:
                key = cb.currentText()
                input_entry = {
                    "key": key,
                    "elements": elem_input.text(),
                    "tri": tri_cb.isChecked(),
                    "plot": plot_cb.isChecked(),
                    "mapping": map_dropdown.currentText()
                }
                
                # Save complete transform configuration if available
                if hasattr(self, 'args') and 'map_input' in self.args and key in self.args['map_input']:
                    input_entry["transform_config"] = self.args['map_input'][key]
                
                state["inputs"].append(input_entry)

            # Save output widgets with complete transform settings and bounds
            for cb, plot_cb, map_dropdown, bounds_cb in self.output_var_cbs:
                key = cb.currentText()
                output_entry = {
                    "key": key,
                    "plot": plot_cb.isChecked(),
                    "map": map_dropdown.currentText(),
                    "bounds": bounds_cb.isChecked()  # store checkbox state
                }
                
                # Save complete transform configuration if available
                if hasattr(self, 'args') and 'map_output' in self.args and key in self.args['map_output']:
                    output_entry["transform_config"] = self.args['map_output'][key]
                
                # Save bounds values if set
                if hasattr(self, 'args') and 'constrain_regression' in self.args and key in self.args['constrain_regression']:
                    output_entry["bounds_values"] = self.args['constrain_regression'][key]
                
                state["outputs"].append(output_entry)

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            print("State saved.")
        except Exception as e:
            print(f"Error saving state: {e}")

        
    def setup_ui(self):
        grid = QtWidgets.QGridLayout()
        self.setup_layout.addLayout(grid)

        # ---- Load and Add/Load More Data buttons on the same row ----
        load_button = QtWidgets.QPushButton("Load Data (.pickle file)")
        load_button.clicked.connect(self.load_data)
        grid.addWidget(load_button, 0, 0, 1, 1)

        add_more_button = QtWidgets.QPushButton("Add/Load More Data")
        add_more_button.clicked.connect(self.add_more_data)
        grid.addWidget(add_more_button, 0, 1, 1, 1)

        # ---- New button: Save Merged Data (below the two buttons) ----
        save_merged_button = QtWidgets.QPushButton("Save Merged Data")
        save_merged_button.clicked.connect(self.save_merged_data)
        grid.addWidget(save_merged_button, 1, 0, 1, 2)  # spans both columns

        # ---- State buttons (moved below Save Merged Data) ----
        state_btn_row = QtWidgets.QHBoxLayout()
        state_btn_row.setAlignment(QtCore.Qt.AlignLeft)  # align to the left

        save_btn = QtWidgets.QPushButton("💾 State")
        save_btn.setFixedWidth(100)
        save_btn.clicked.connect(self.save_state_to_file)

        load_btn = QtWidgets.QPushButton("📂 State")
        load_btn.setFixedWidth(100)
        load_btn.clicked.connect(self.load_state_from_file)

        state_btn_row.addWidget(save_btn)
        state_btn_row.addWidget(load_btn)

        # Place the state buttons on the next row and span both columns
        grid.addLayout(state_btn_row, 2, 0, 1, 2)

        # ---- File path label (moved down) ----
        self.file_path_label = QtWidgets.QLabel("No file loaded")
        self.file_path_label.setStyleSheet("color: gray; font-size: 10px;")
        self.file_path_label.setWordWrap(True)
        grid.addWidget(self.file_path_label, 3, 0, 1, 2)  # Spans 2 columns
        # Keep track of loaded files for concise display
        self.loaded_files = []
        

        self.input_container = QtWidgets.QVBoxLayout()
        self.input_container.setSpacing(5)
        self.input_container.setContentsMargins(0, 0, 0, 0)

        self.output_container = QtWidgets.QVBoxLayout()
        self.output_container.setSpacing(5)
        self.output_container.setContentsMargins(0, 0, 0, 0)

        self.input_group = QtWidgets.QGroupBox("Input Variables")
        self.input_group.setLayout(self.input_container)
        self.setup_layout.addWidget(self.input_group)

        # Create a horizontal layout row
        input_and_mesh_row = QtWidgets.QHBoxLayout()

        # Add the input buttons (aligned left)
        remove_input_button = QtWidgets.QPushButton("➖")
        remove_input_button.setFixedSize(20, 16)
        remove_input_button.clicked.connect(self.remove_input_cb)

        add_input_button = QtWidgets.QPushButton("➕")
        add_input_button.setFixedSize(20, 16)
        add_input_button.clicked.connect(self.add_input_cb)

        input_and_mesh_row.addWidget(remove_input_button)
        input_and_mesh_row.addWidget(add_input_button)

        # Add a stretch in between to push the next button to the right
        input_and_mesh_row.addStretch(1)


        # Add "Preview Mesh" button
        self.preview_mesh_btn = QtWidgets.QPushButton("Preview Mesh")
        self.preview_mesh_btn.setFixedWidth(120)
        self.preview_mesh_btn.setStyleSheet("background-color: orange; color: black;")
        self.preview_mesh_btn = QtWidgets.QPushButton("Preview Mesh")
        self.preview_mesh_btn.clicked.connect(self.preview_mesh_overlay)
        input_and_mesh_row.addWidget(self.preview_mesh_btn)

        self.remove_highlight_btn = QtWidgets.QPushButton("Remove Highlighted Elements")
        self.remove_highlight_btn.clicked.connect(self.remove_highlighted_elements)
        self.remove_highlight_btn.setFixedWidth(180)
        input_and_mesh_row.addWidget(self.remove_highlight_btn)

        # Finally, add this row to the setup layout
        self.setup_layout.addLayout(input_and_mesh_row)

        self.refine_region_btn = QtWidgets.QPushButton("Refine Region")
        self.refine_region_btn.setFixedWidth(150)
        self.refine_region_btn.clicked.connect(self.trigger_lasso_selector)
        input_and_mesh_row.addWidget(self.refine_region_btn)

        
        self.output_group = QtWidgets.QGroupBox("Output Variables")
        self.output_group.setLayout(self.output_container)
        self.setup_layout.addWidget(self.output_group)

        output_btn_row = QtWidgets.QHBoxLayout()
        output_btn_row.setAlignment(QtCore.Qt.AlignLeft)

        remove_output_button = QtWidgets.QPushButton("➖")
        remove_output_button.setFixedSize(20, 16)
        remove_output_button.clicked.connect(self.remove_output_cb)

        add_output_button = QtWidgets.QPushButton("➕")
        add_output_button.setFixedSize(20, 16)
        add_output_button.clicked.connect(self.add_output_cb)

        output_btn_row.addWidget(remove_output_button)
        output_btn_row.addWidget(add_output_button)
        self.setup_layout.addLayout(output_btn_row)


        reset_button = QtWidgets.QPushButton("Reset to Default")
        reset_button.clicked.connect(self.reset_inputs_outputs)
        self.setup_layout.addWidget(reset_button)

        preview_button = QtWidgets.QPushButton("Review Configs")
        preview_button.clicked.connect(self.preview_dicts_only)
        self.setup_layout.addWidget(preview_button)

        self.toggle_data_plot_btn = QtWidgets.QPushButton("Switch 3D View")
        self.toggle_data_plot_btn.clicked.connect(self.toggle_data_plot_view)
        self.toggle_data_plot_btn.setFixedWidth(200)
        self.data_tab_layout.addWidget(self.toggle_data_plot_btn)

        self.draw_roi_button = QtWidgets.QPushButton("Draw ROI")
        self.draw_roi_button.setCheckable(True)
        self.draw_roi_button.setFixedWidth(200)
        self.draw_roi_button.setToolTip("Activate ROI selection on the mesh")
        self.draw_roi_button.toggled.connect(self.toggle_draw_roi)
        self.data_tab_layout.addWidget(self.draw_roi_button)

        self.manual_roi_btn = QtWidgets.QPushButton("ROI Corrections")
        self.manual_roi_btn.setFixedWidth(200)
        self.manual_roi_btn.clicked.connect(self.open_manual_roi_dialog)
        self.data_tab_layout.addWidget(self.manual_roi_btn)


        self.toggle_mapped_output_in_data_preview = QtWidgets.QCheckBox("Logarithmic colors")
        self.toggle_mapped_output_in_data_preview.setChecked(False)
        self.toggle_mapped_output_in_data_preview.stateChanged.connect(self.plot_data)
        self.data_tab_layout.addWidget(self.toggle_mapped_output_in_data_preview)

        self.log_x_checkbox = QtWidgets.QCheckBox("Log X-axis")
        self.log_x_checkbox.setChecked(False)
        self.log_x_checkbox.stateChanged.connect(self.plot_data)
        self.data_tab_layout.addWidget(self.log_x_checkbox)

        self.log_y_checkbox = QtWidgets.QCheckBox("Log Y-axis")
        self.log_y_checkbox.setChecked(False)
        self.log_y_checkbox.stateChanged.connect(self.plot_data)
        self.data_tab_layout.addWidget(self.log_y_checkbox)


        plot_button = QtWidgets.QPushButton("Plot Data")
        plot_button.clicked.connect(self.plot_data)
        self.setup_layout.addWidget(plot_button)

        run_button = QtWidgets.QPushButton("Run Training")
        run_button.clicked.connect(self.run_training)
        self.setup_layout.addWidget(run_button)

        # Number of result 3D plot points spinbox
        self.num_points_spinbox = QtWidgets.QSpinBox()
        self.num_points_spinbox.setRange(10, 10000)        
        self.num_points_spinbox.setValue(200)             # default 200
        self.num_points_spinbox.setSingleStep(50)
        self.num_points_spinbox.setToolTip("Number of random points to draw in 3D view")
        self.num_points_spinbox.valueChanged.connect(self.update_result_plot)  # replot on change

        label = QtWidgets.QLabel("3D Points:")
        points_row = QtWidgets.QHBoxLayout()
        points_row.setSpacing(4)  # Reduce horizontal space between label and spinbox
        points_row.addWidget(label)
        # points_row.addStretch()  # prevents the label and spinbox from stretching across the domain, and stay close to each other
        points_row.addWidget(self.num_points_spinbox)
        self.result_options_layout.addLayout(points_row)
        
        self.preview_points_spinbox = QtWidgets.QSpinBox()
        self.preview_points_spinbox.setRange(10, 1000000)
        self.preview_points_spinbox.setSingleStep(100)
        self.preview_points_spinbox.setFixedWidth(200)
        self.preview_points_spinbox.setValue(10000)
        self.preview_points_spinbox.setToolTip("Number of random data points to show in 3D view")
        self.preview_points_spinbox.valueChanged.connect(self.plot_data)

        preview_label = QtWidgets.QLabel("Preview Points:")
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.setSpacing(4)
        preview_row.addStretch() # prevents the label and spinbox from stretching across the domain, and stay close to each other
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.addWidget(preview_label)
        preview_row.addWidget(self.preview_points_spinbox)

        preview_widget = QtWidgets.QWidget()
        preview_widget.setLayout(preview_row)
        self.data_tab_layout.addWidget(preview_widget)


    def trigger_lasso_selector(self):
        self.enable_lasso_selector = True
        ax2d = self.data_canvas_2d.figure.axes[0]  # or wherever your 2D plot is
        if hasattr(self, "lasso") and self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        self.lasso = LassoSelector(ax2d, onselect=self.on_lasso_select)
        self.enable_lasso_selector = False


    def open_manual_roi_dialog(self):
        input_vars = list(self.mesh_specs["element_numbers"].keys())
        var_name, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Input Variable", "Select variable to adjust ROI:", input_vars, 0, False)
        if not ok or var_name == "":
            return

        # Get current min/max if any
        current_min, current_max = None, None
        if hasattr(self, "roi") and var_name in self.roi:
            current_min, current_max = self.roi[var_name]
        else:
            # fallback to auto min/max
            current_min = np.min(self.data["data"][var_name])
            current_max = np.max(self.data["data"][var_name])

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Set ROI for {var_name}")
        layout = QtWidgets.QVBoxLayout()

        min_input = QtWidgets.QLineEdit(str(current_min))
        max_input = QtWidgets.QLineEdit(str(current_max))

        layout.addWidget(QtWidgets.QLabel("Min:"))
        layout.addWidget(min_input)
        layout.addWidget(QtWidgets.QLabel("Max:"))
        layout.addWidget(max_input)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            try:
                new_min = float(min_input.text())
                new_max = float(max_input.text())
                if not hasattr(self, "roi"):
                    self.roi = {}
                self.roi[var_name] = (new_min, new_max)
                self._normalize_roi()
                self.build_dicts(update_text=True)  # Refresh previews etc.
                QtWidgets.QMessageBox.information(self, "ROI Updated", f"ROI for '{var_name}' updated.")
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", "Min/Max must be valid floats.")

    def toggle_draw_roi(self, checked):
        if not hasattr(self, 'data_canvas_2d') or self.data_canvas_2d is None:
            print("No canvas to attach ROI to.")
            return

        ax = self.data_canvas_2d.figure.axes[0]  # Assumes only one axes
        if checked:
            self.start_roi_selector(ax)
        else:
            if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
                self.rectangle_selector.set_active(False)
                self.rectangle_selector.disconnect_events()
                self.rectangle_selector = None
                self.data_canvas_2d.draw()

    def setup_interactive_labels(self, fig):
        """Set up interactive label editing for a figure"""
        def on_click(event):
            if event.button != 1:  # Left click only
                return
                
            # Check if we clicked on any text artists
            for text in fig.findobj(match=lambda x: isinstance(x, plt.Text)):
                # Skip if this is a tick label
                if hasattr(text, '_text') and (isinstance(text._text, str) and text._text.startswith('\u2212')):
                    continue
                    
                # Check if the click was on this text object
                if text.contains(event)[0]:
                    # Only edit text objects that have actual content
                    if text.get_text() and not text in self.editable_labels:
                        # Store reference to the text object and its current text
                        self.editable_labels[text] = text.get_text()
                        try:
                            self.edit_label(text)
                        except Exception as e:
                            print(f"Error editing label: {e}")
                    break
        
        # Connect to the figure's canvas
        fig.canvas.mpl_connect('button_press_event', on_click)
        
    def edit_label(self, text_obj):
        """Open a dialog to edit the text of a label"""
        current_text = text_obj.get_text()
        
        # Create a simple dialog with a text input
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Edit Label")
        layout = QtWidgets.QVBoxLayout()
        
        # Determine what type of label this is
        label_type = "Label"
        if hasattr(text_obj, '_label_type'):
            label_type = text_obj._label_type
        elif hasattr(text_obj, 'axes') and text_obj.axes is not None:
            # Now check axes attributes only if axes exists
            if hasattr(text_obj.axes, 'xaxis') and text_obj is text_obj.axes.xaxis.get_label():
                label_type = "X-Axis Label"
            elif hasattr(text_obj.axes, 'yaxis') and text_obj is text_obj.axes.yaxis.get_label():
                label_type = "Y-Axis Label"
            elif (hasattr(text_obj.axes, 'zaxis') and 
                text_obj.axes.zaxis is not None and 
                text_obj is text_obj.axes.zaxis.get_label()):
                label_type = "Z-Axis Label"
            elif hasattr(text_obj.axes, 'get_title') and text_obj == text_obj.axes.get_title():
                label_type = "Title"
            
        info_label = QtWidgets.QLabel(f"Edit {label_type}:")
        layout.addWidget(info_label)
        
        text_edit = QtWidgets.QLineEdit(current_text)
        layout.addWidget(text_edit)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        # If dialog is accepted, update the text
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_text = text_edit.text()
            text_obj.set_text(new_text)
            text_obj.figure.canvas.draw()
    
    def toggle_theme(self, state):
        """Toggle between dark and light theme for plots"""
        is_dark = bool(state == QtCore.Qt.CheckState.Checked)
        smp.darktheme(is_dark)
        
        # Update all existing plots
        for canvas in [self.data_canvas_2d, self.data_canvas_3d, 
                      self.pca_canvas, self.ts_canvas]:
            if canvas is not None:
                fig = canvas.figure
                self._apply_theme_to_figure(fig, is_dark)
                canvas.draw()
    
    def _apply_theme_to_figure(self, fig, is_dark):
        """Apply the selected theme to a matplotlib figure"""
        if is_dark:
            bg_color = '#1e1e1e'
            text_color = 'white'
            grid_color = '#555555'
        else:
            bg_color = 'white'
            text_color = 'black'
            grid_color = '#cccccc'
        
        # Apply theme to figure and all axes
        fig.patch.set_facecolor(bg_color)
        for ax in fig.get_axes():
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            if hasattr(ax, 'zaxis'):
                ax.zaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['top'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            ax.spines['right'].set_color(text_color)
            
            # Apply to grid if present
            if ax.get_xgridlines() or ax.get_ygridlines():
                ax.grid(color=grid_color, linestyle='-', linewidth=0.5, alpha=0.5)
                
            # Apply to any collections (e.g., contour plots)
            for collection in ax.collections:
                if hasattr(collection, 'set_edgecolor'):
                    collection.set_edgecolor(text_color)
            
            # Apply to legend if present
            if ax.get_legend():
                ax.get_legend().get_frame().set_facecolor(bg_color)
                ax.get_legend().get_frame().set_edgecolor(text_color)
                for text in ax.get_legend().get_texts():
                    text.set_color(text_color)
                    
        # Apply to colorbar(s) if present
        for child in fig.get_children():
            if hasattr(child, 'ax') and hasattr(child, 'outline'):
                # This is likely a colorbar
                cb = child
                if hasattr(cb.ax, 'yaxis') and hasattr(cb.ax.yaxis, 'label'):
                    cb.ax.yaxis.label.set_color(text_color)
                cb.ax.tick_params(colors=text_color)
                cb.outline.set_edgecolor(text_color)
            
    def toggle_data_plot_view(self):
        if self.data_plot_stack.currentWidget() == self.data_canvas_2d and self.data_canvas_3d:
            self.data_plot_stack.setCurrentWidget(self.data_canvas_3d)
        elif self.data_canvas_2d:
            self.data_plot_stack.setCurrentWidget(self.data_canvas_2d)

    def build_state_dict(self):
        state = {
            "input_vars": [],
            "output_vars": [],
            "plot_2d_mode": self.plot_2d_mode,
            "args": self.args,  # optional, if needed
        }
        for cb, elem_input, tri_cb, plot_cb, map_dropdown, _ in self.input_var_widgets:
            state["input_vars"].append({
                "key": cb.currentText(),
                "elem": elem_input.text(),
                "tri": tri_cb.isChecked(),
                "plot": plot_cb.isChecked(),
                "map": map_dropdown.currentText()
            })
        # #OLD: before bounds button
        # for cb, plot_cb, map_dropdown, bounds, in self.output_var_cbs:
        #     state["output_vars"].append({
        #         "key": cb.currentText(),
        #         "plot": plot_cb.isChecked(),
        #         "map": map_dropdown.currentText()
        #     })

        # NEW
        for cb, plot_cb, map_dropdown, bounds_btn in self.output_var_cbs:
            var_name = cb.currentText()
            var_entry = {
                "key": var_name,
                "plot": plot_cb.isChecked(),
                "map": map_dropdown.currentText()
            }
            # Add bounds if they were set
            if hasattr(cb, "bounds_values") and cb.bounds_values is not None:
                var_entry["bounds"] = cb.bounds_values  # {"lowerbound": ..., "upperbound": ...}


            state["output_vars"].append(var_entry)

        return state

    def apply_state_dict(self, state):
        self.reset_inputs_outputs()

        for input_config in state.get("input_vars", []):
            self.add_input_cb()
            cb, elem_input, tri_cb, plot_cb, map_dropdown, _ = self.input_var_widgets[-1]
            cb.setCurrentText(input_config["key"])
            elem_input.setText(input_config["elem"])
            tri_cb.setChecked(input_config["tri"])
            plot_cb.setChecked(input_config["plot"])
            map_dropdown.setCurrentText(input_config["map"])

        for output_config in state.get("output_vars", []):
            self.add_output_cb()
            cb, plot_cb, map_dropdown = self.output_var_cbs[-1]
            cb.setCurrentText(output_config["key"])
            plot_cb.setChecked(output_config["plot"])
            map_dropdown.setCurrentText(output_config["map"])

        self.args = state.get("args", {})
        self.update_dict_preview()


    def save_state_to_file(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save GUI State", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            state = self.build_state_dict()
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Success", "State saved successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save state:\n{str(e)}")

    # def load_state_from_file(self):
    #     file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
    #         self, "Load GUI State", "", "JSON Files (*.json)"
    #     )
    #     if not file_path:
    #         return

    #     try:
    #         with open(file_path, 'r') as f:
    #             state = json.load(f)
    #         self.apply_state_dict(state)
    #         QtWidgets.QMessageBox.information(self, "Success", "State loaded successfully.")
    #     except Exception as e:
    #         QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load state:\n{str(e)}")

    def load_state_from_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load GUI State", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                state = json.load(f)

            # Apply state to GUI
            self.reset_inputs_outputs()  # clear current widgets
            self.roi = state.get("roi", None)
            self.mesh_specs = state.get("mesh_specs", {})
            self.args = state.get("args", {})

             # Rebuild input widgets
            for input_cfg in state.get("input_vars", []):  # was "inputs"
                self.add_input_cb()
                cb, elem_input, tri_cb, plot_cb, map_dropdown, scaler_btn = self.input_var_widgets[-1]
                cb.setCurrentText(input_cfg.get("key", ""))
                elem_input.setText(input_cfg.get("elem", ""))
                tri_cb.setChecked(input_cfg.get("tri", False))
                plot_cb.setChecked(input_cfg.get("plot", False))
                mapping_type = input_cfg.get("map", "minmax")
                idx = map_dropdown.findText(mapping_type)
                if idx >= 0:
                    map_dropdown.setCurrentIndex(idx)

            # Rebuild output widgets
            for out_cfg in state.get("output_vars", []):  # was "outputs"
                self.add_output_cb()
                cb, plot_cb, map_dropdown, bounds_cb = self.output_var_cbs[-1]
                cb.setCurrentText(out_cfg.get("key", ""))
                plot_cb.setChecked(out_cfg.get("plot", False))
                map_text = out_cfg.get("map", "minmax")
                idx = map_dropdown.findText(map_text)
                if idx >= 0:
                    map_dropdown.setCurrentIndex(idx)

                # Bounds: check if there is a "bounds" dict
                bounds = out_cfg.get("bounds", None)
                if bounds is not None:
                    bounds_cb.setChecked(True)
                    cb.bounds_values = bounds  # store bounds for later
                else:
                    bounds_cb.setChecked(False)
                    cb.bounds_values = None

            # # Rebuild input widgets
            # for input_cfg in state.get("inputs", []):
            #     self.add_input_cb()
            #     cb, elem_input, tri_cb, plot_cb, map_dropdown, scaler_btn = self.input_var_widgets[-1]
            #     cb.setCurrentText(input_cfg.get("key", ""))
            #     elem_input.setText(input_cfg.get("elements", ""))
            #     tri_cb.setChecked(input_cfg.get("tri", False))
            #     plot_cb.setChecked(input_cfg.get("plot", False))
            #     mapping_type = input_cfg.get("mapping", "minmax")
            #     idx = map_dropdown.findText(mapping_type)
            #     if idx >= 0:
            #         map_dropdown.setCurrentIndex(idx)

            # # Rebuild output widgets
            # for out_cfg in state.get("outputs", []):
            #     self.add_output_cb()
            #     cb, plot_cb, map_dropdown, bounds_cb = self.output_var_cbs[-1]
            #     cb.setCurrentText(out_cfg.get("key", ""))
            #     plot_cb.setChecked(out_cfg.get("plot", False))
            #     map_text = out_cfg.get("map", "minmax")
            #     idx = map_dropdown.findText(map_text)
            #     if idx >= 0:
            #         map_dropdown.setCurrentIndex(idx)
            #     bounds_cb.setChecked(out_cfg.get("bounds", False))

            QtWidgets.QMessageBox.information(self, "Success", "State loaded successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load state:\n{str(e)}")


    def add_input_cb(self):
        container = QtWidgets.QHBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.setSpacing(5)

        cb = QtWidgets.QComboBox()
        if self.input_keys:
            cb.addItems(self.input_keys)

        elem_input = QtWidgets.QLineEdit("6")
        elem_input.setFixedWidth(50)  # or 50 if you want a little more room
        tri_cb = QtWidgets.QCheckBox("Tri")
        plot_cb = QtWidgets.QCheckBox()
        plot_cb.setToolTip("Select variable for plotting (max 2)")

        # Scaler dropdown
        scaler_cb = QtWidgets.QComboBox()
        scaler_cb.addItems(["minmax", "log10", "compress", "symlog"])
        scaler_cb.setToolTip("Select mapping/scaling method")

        # Scaler settings button (⚙ cogwheel)
        scaler_btn = QtWidgets.QPushButton("⚙")
        scaler_btn.setFixedWidth(30)
        scaler_btn.setToolTip("Open scaler settings")

        # Connect plot toggle limit
        plot_cb.toggled.connect(self.enforce_max_two_plot_inputs)

        # Connect scaler button to settings dialog
        # scaler_btn.clicked.connect(lambda _, combo=scaler_cb, key_cb=cb: self.open_scaler_settings(combo, key_cb))
        scaler_btn.clicked.connect(lambda _, combo=scaler_cb, key_cb=cb: self.open_scaler_settings(combo, key_cb, is_input=True))

        # Add widgets to layout
        container.addWidget(cb)
        container.addWidget(QtWidgets.QLabel("Elements:"))
        container.addWidget(elem_input)
        container.addWidget(tri_cb)
        container.addWidget(QtWidgets.QLabel("Scaler:"))
        container.addWidget(scaler_cb)
        container.addWidget(scaler_btn)
        plot_cb = QtWidgets.QCheckBox("Plot")
        container.addWidget(plot_cb)

        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(container)
        self.input_container.addWidget(wrapper)

        # Store all 6 widgets
        self.input_var_widgets.append((cb, elem_input, tri_cb, plot_cb, scaler_cb, scaler_btn))


    def open_scaler_settings(self, scaler_cb, var_cb, is_input):
        var_name = var_cb.currentText()
        scaler_type = scaler_cb.currentText()
        map_key = "map_input" if is_input else "map_output"

        # Get existing config
        current_config = self.args.get(map_key, {}).get(var_name, {}).get(scaler_type, {})

        dialog = ScalerSettingsDialog(var_name, scaler_type, current_config)
        if dialog.exec_():
            updated_config = dialog.get_config()
            if map_key not in self.args:
                self.args[map_key] = {}
            if var_name not in self.args[map_key]:
                self.args[map_key][var_name] = {}

            self.args[map_key][var_name][scaler_type] = updated_config
            print(f"Updated {map_key} for {var_name}: {self.args[map_key][var_name]}")

    def enforce_max_two_plot_inputs(self):
        # Count how many plot checkboxes are checked
        checked = [w[3] for w in self.input_var_widgets if w[3].isChecked()]
        if len(checked) > 2:
            # Uncheck the one that was just checked (sender)
            sender = self.sender()
            if sender.isChecked():
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)
                QtWidgets.QMessageBox.warning(self, "Plot Selection", "Please select only two input variables for plotting.")

    def remove_input_cb(self):
        if self.input_var_widgets:
            widget = self.input_var_widgets.pop()
            widget[0].parent().setParent(None)

    def reset_inputs_outputs(self):
        # Clear input widgets
        for i in reversed(range(self.input_container.count())):
            self.input_container.itemAt(i).widget().setParent(None)
        self.input_var_widgets = []

        # Clear output widgets
        for i in reversed(range(self.output_container.count())):
            self.output_container.itemAt(i).widget().setParent(None)
        self.output_var_cbs = []

 
    # # OLD
    # def add_output_cb(self):
    #     container = QtWidgets.QHBoxLayout()
    #     container.setContentsMargins(0, 0, 0, 0)
    #     container.setSpacing(5)

    #     cb = QtWidgets.QComboBox()
    #     if self.output_keys:
    #         cb.addItems(self.output_keys)

    #     scaler_dropdown = QtWidgets.QComboBox()
    #     scaler_dropdown.addItems(["minmax", "log10", "compress"])

    #     scaler_btn = QtWidgets.QPushButton("⚙")
    #     scaler_btn.setFixedWidth(25)

    #     # 🔧 Connect dropdown change to open settings
    #     scaler_dropdown.currentIndexChanged.connect(
    #         lambda idx, scb=scaler_dropdown, kcb=cb: self.open_scaler_settings(scb, kcb, is_input=False)
    #     )

    #     # Keep existing ⚙ button behavior
    #     # scaler_btn.clicked.connect(lambda _, combo=scaler_dropdown, key_cb=cb: self.open_scaler_settings(combo, key_cb))
    #     scaler_btn.clicked.connect(lambda _, combo=scaler_dropdown, key_cb=cb: self.open_scaler_settings(combo, key_cb, is_input=False))
    #     container.addWidget(cb)
    #     container.addWidget(QtWidgets.QLabel("Scaler:"))
    #     container.addWidget(scaler_dropdown)
    #     container.addWidget(scaler_btn)
    #     plot_cb = QtWidgets.QCheckBox("Plot")
    #     container.addWidget(plot_cb)

    #     wrapper = QtWidgets.QWidget()
    #     wrapper.setLayout(container)
    #     self.output_container.addWidget(wrapper)

    #     self.output_var_cbs.append((cb, plot_cb, scaler_dropdown))

    def add_output_cb(self):
        container = QtWidgets.QHBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.setSpacing(5)

        cb = QtWidgets.QComboBox()
        if self.output_keys:
            cb.addItems(self.output_keys)

        # Scaler dropdown
        scaler_dropdown = QtWidgets.QComboBox()
        scaler_dropdown.addItems(["minmax", "log10", "compress", "symlog"])

        # Scaler settings button
        scaler_btn = QtWidgets.QPushButton("⚙")
        scaler_btn.setFixedWidth(35)
        scaler_btn.clicked.connect(lambda _, combo=scaler_dropdown, key_cb=cb: self.open_scaler_settings(combo, key_cb, is_input=False))

        # # New: bounds settings button
        # bounds_btn = QtWidgets.QPushButton("⚙")
        # bounds_btn.setFixedWidth(25)
        # bounds_btn.setToolTip("Set lower and upper nodal value bounds for this output variable")
        # bounds_btn.clicked.connect(lambda _, key_cb=cb: self.open_bounds_dialog(key_cb))

        # New: bounds checkbox
        bounds_cb = QtWidgets.QCheckBox("Constrain")
        bounds_cb.setToolTip("Enable lower/upper bounds for this output variable")

        # Connect checkbox to open bounds dialog only when checked
        def on_bounds_checked(state, key_cb=cb):
            if state == QtCore.Qt.Checked:
                # Only open dialog if bounds values don't already exist
                if not hasattr(key_cb, "bounds_values") or key_cb.bounds_values is None:
                    self.open_bounds_dialog(key_cb)
                else:
                    # Make sure values are in args too
                    var_name = key_cb.currentText()
                    if "constrain_regression" not in self.args:
                        self.args["constrain_regression"] = {}
                    if var_name not in self.args["constrain_regression"] or self.args["constrain_regression"][var_name] is None:
                        self.args["constrain_regression"][var_name] = key_cb.bounds_values
                    print(f"Using existing bounds for {var_name}: {key_cb.bounds_values}")


        bounds_cb.stateChanged.connect(on_bounds_checked)


        # Plot checkbox
        plot_cb = QtWidgets.QCheckBox("Plot")


        # Add widgets to layout
        container.addWidget(cb)
        container.addWidget(QtWidgets.QLabel("Scaler:"))
        container.addWidget(scaler_dropdown)
        container.addWidget(scaler_btn)
        container.addWidget(bounds_cb)  # <-- add bounds button here
        container.addWidget(plot_cb)

        # Wrap in QWidget and add to output container
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(container)
        self.output_container.addWidget(wrapper)

        # Save widgets
        self.output_var_cbs.append((cb, plot_cb, scaler_dropdown, bounds_cb))


    def open_bounds_dialog(self, var_cb):
        var_name = var_cb.currentText()
        current_config = self.args.get("constrain_regression", {}).get(var_name)
        lower, upper = None, None
        if isinstance(current_config, dict):
            lower = current_config.get("lowerbound", None)
            upper = current_config.get("upperbound", None)

        dialog = BoundsSettingsDialog(var_name, lower, upper)
        if dialog.exec_():
            lower_val, upper_val = dialog.get_bounds()
            if "constrain_regression" not in self.args:
                self.args["constrain_regression"] = {}
            self.args["constrain_regression"][var_name] = {
                "lowerbound": lower_val,
                "upperbound": upper_val
            }
            # Store bounds values directly on combobox for later retrieval
            var_cb.bounds_values = {
                "lowerbound": lower_val,
                "upperbound": upper_val
            }
            print(f"Updated constrain_regression for {var_name}: {self.args['constrain_regression'][var_name]}")


    def remove_output_cb(self):
        if self.output_var_cbs:
            widget = self.output_var_cbs.pop()
            widget[0].parent().setParent(None)

    def load_data_from_file(self, file_path, reset_ui=True):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

        self._clean_legacy_data()

        # Clear previous mesh specs and args before rebuilding
        self.mesh_specs = {}
        self.args = {}
        self.args["data_file"] = file_path
    
        # Update the list of loaded files and refresh the label
        self.loaded_files = [file_path]  # start fresh with the newly loaded file
        self._update_file_path_label()

        self.input_keys = [k for k in self.data['data'].keys() if k not in ('U', 'x')]
        self.output_keys = list(self.data['data']['U'].keys())

        # Only reset and rebuild UI widgets if requested
        if reset_ui:
            self.reset_inputs_outputs()
            for cb, *_ in self.input_var_widgets:
                cb.clear()
                cb.addItems(self.input_keys)
            for cb, _ in self.output_var_cbs:
                cb.clear()
                cb.addItems(self.output_keys)
        else:
            # Just update combo box options without resetting UI layout
            for cb, *_ in self.input_var_widgets:
                cb.clear()
                cb.addItems(self.input_keys)
            for cb, _ in self.output_var_cbs:
                cb.clear()
                cb.addItems(self.output_keys)

        
    def load_data(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Pickle File", "", "Pickle Files (*.pkl *.pickle)"
        )
        if not file_path:
            return

        self.load_data_from_file(file_path, reset_ui=True)

        self._clean_legacy_data()

        # Clear previous mesh specs and args before rebuilding
        self.mesh_specs = {}
        self.args = {}
        self.roi = None
        self.preview_nodes = None
        self.preview_conn = None
        self.preview_mesh = True  # Force a clean preview

        # Save data file path in args
        self.args["data_file"] = file_path

        # Extract input and output keys
        self.input_keys = [k for k in self.data['data'].keys() if k != 'U']
        self.output_keys = list(self.data['data']['U'].keys())

        # Clear previous input/output widgets
        self.reset_inputs_outputs()

        # Add input selectors: two by default or as many as available (min 2)
        if self.input_keys:
            for i in range(min(2, len(self.input_keys))):
                self.add_input_cb()
                cb, elem_input, tri_cb, plot_cb, _, _ = self.input_var_widgets[-1]
                cb.setCurrentText(self.input_keys[i])
                elem_input.setText("6")  # default elements count
                tri_cb.setChecked(False)
                plot_cb.setChecked(False)
        else:
            for _ in range(2):
                self.add_input_cb()

        # Add one output selector by default
        if self.output_keys:
            self.add_output_cb()
            cb, _, map_dropdown, _ = self.output_var_cbs[-1]
            cb.setCurrentText(self.output_keys[0])
            map_dropdown.setCurrentText("minmax")  # default mapping
        else:
            self.add_output_cb()


    def add_more_data(self):
        """Open a file dialog to load an additional dataset and merge it with the
        currently loaded data, handling missing columns by prompting for a fill value."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Additional Pickle File", "", "Pickle Files (*.pkl *.pickle)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'rb') as f:
                new_data = pickle.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error",
                                        f"Failed to load the selected file:\\n{e}")
            return

        # Determine missing columns
        existing_inputs = set(k for k in self.data['data'].keys() if k not in ('U', 'x'))
        new_inputs = set(k for k in new_data['data'].keys() if k not in ('U', 'x'))

        existing_outputs = set(self.data['data']['U'].keys())
        new_outputs = set(new_data['data']['U'].keys())

        missing_in_existing = new_inputs.union(new_outputs) - existing_inputs.union(existing_outputs)
        missing_in_new = existing_inputs.union(existing_outputs) - new_inputs.union(new_outputs)

        fill_value = 0.0
        if missing_in_existing or missing_in_new:
            missing_info = []
            if missing_in_existing:
                missing_info.append("Columns missing in the currently loaded data:")
                missing_info.extend(f"  - {col}" for col in sorted(missing_in_existing))
            if missing_in_new:
                missing_info.append("Columns missing in the newly loaded data:")
                missing_info.extend(f"  - {col}" for col in sorted(missing_in_new))

            # Use the custom MissingColumnsDialog which contains a plain QLineEdit.
            # This allows users to type any string that Python's float() can parse,
            # including exponential notation such as "1e-9".
            dlg = MeshSettingsDialog.MissingColumnsDialog("\n".join(missing_info), parent=self)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                try:
                    fill_value = float(dlg.get_fill_value())
                except ValueError:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Fill",
                        "Fill value must be a number. Using default 0.0."
                    )
                    fill_value = 0.0
            else:
                # User cancelled the dialog – abort the merge.
                return

        # Consolidate data
        self._consolidate_additional_data(new_data, fill_value)

        # Update the list of loaded files and refresh the label
        if not hasattr(self, "loaded_files"):
            self.loaded_files = []
        self.loaded_files.append(file_path)
        # Keep only the most recent 5 files to avoid overly long labels
        self.loaded_files = self.loaded_files[-5:]
        self._update_file_path_label()

        # Recalculate ROI based on the merged dataset
        self.build_dicts(update_text=True)   # recompute ROI and refresh dict preview
        # Ensure ROI includes all current input dimensions
        self.roi = self.get_roi_from_inputs(self.input_keys, self.data)
        self.update_dict_preview()

        # Invalidate any existing preview mesh so that it will be rebuilt with the new ROI
        self.preview_nodes = None
        self.preview_conn = None
        if hasattr(self, 'mesh_overlay_artists'):
            self.mesh_overlay_artists = []

    def _consolidate_additional_data(self, new_data, fill_value=0.0):
        """Merge ``new_data`` into ``self.data`` ensuring both have the same columns."""
        def filler(length):
            return np.full(length, fill_value, dtype=float)

        existing_len = len(next(iter(self.data['data'].values()))) if self.data['data'] else 0
        new_len = len(next(iter(new_data['data'].values()))) if new_data['data'] else 0

        all_input_cols = set(k for k in self.data['data'].keys() if k not in ('U', 'x')).union(
            set(k for k in new_data['data'].keys() if k not in ('U', 'x'))
        )
        all_output_cols = set(self.data['data']['U'].keys()).union(set(new_data['data']['U'].keys()))

        merged = {'data': {}}
        for col in all_input_cols:
            arr_existing = np.asarray(self.data['data'].get(col, filler(existing_len)))
            arr_new = np.asarray(new_data['data'].get(col, filler(new_len)))
            merged['data'][col] = np.concatenate([arr_existing, arr_new])

        merged['data']['U'] = {}
        for col in all_output_cols:
            arr_existing = np.asarray(self.data['data']['U'].get(col, filler(existing_len)))
            arr_new = np.asarray(new_data['data']['U'].get(col, filler(new_len)))
            merged['data']['U'][col] = np.concatenate([arr_existing, arr_new])

        self.data = merged
        self.input_keys = [k for k in self.data['data'].keys() if k not in ('U', 'x')]
        self.output_keys = list(self.data['data']['U'].keys())

        for cb, *_ in self.input_var_widgets:
            cur = cb.currentText()
            cb.clear()
            cb.addItems(self.input_keys)
            if cur in self.input_keys:
                cb.setCurrentText(cur)

        for cb, *_ in self.output_var_cbs:
            cur = cb.currentText()
            cb.clear()
            cb.addItems(self.output_keys)
            if cur in self.output_keys:
                cb.setCurrentText(cur)

    def _clean_legacy_data(self):
        # Fix old datasets with 'unmapped' key
        if "unmapped" in self.data["data"]:
            if "t" in self.data["data"]:
                time_tmp = self.data["data"]["t"]
                self.data["data"] = self.data["data"]["unmapped"]
                self.data["data"]["t"] = time_tmp
            else:
                self.data["data"] = self.data["data"]["unmapped"]

        # Remove 'dt' if present
        self.data["data"].pop("dt", None)

        # Patch metadata if 'job_number' is missing
        if "meta_data" in self.data:
            if "job_number" not in self.data["meta_data"]:
                self.data["meta_data"]["job_number"] = list(
                    np.arange(0, len(self.data["meta_data"]["sim_lengths"]))
                )

    def plot_data(self):
        """
        Plot the data points
        """
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        selected_inputs = [w for w in self.input_var_widgets if w[3].isChecked()]
        n_selected = len(selected_inputs)

        if n_selected == 0 or n_selected > 2:
            QtWidgets.QMessageBox.warning(self, "Input Selection",
                                        "Please select 1 or 2 input variables to plot.")
            return

        # Call build_dicts to parse inputs, set elements, and ensure ROI
        self.build_dicts(update_text=True)

        var1 = selected_inputs[0][0].currentText()
        var2 = selected_inputs[1][0].currentText() if n_selected == 2 else None

        selected_output = None
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                selected_output = cb.currentText()
                break

        if not selected_output:
            QtWidgets.QMessageBox.warning(self, "No Output Selected",
                                        "Please select an output variable to plot.")
            return

        outvar = selected_output
        max_points = self.preview_points_spinbox.value()
        num_points = len(self.data['data'][var1])
        indices = np.random.choice(num_points, min(max_points, num_points), replace=False)

        # Remove previous canvases
        for canvas in [self.data_canvas_2d, self.data_canvas_3d]:
            if canvas:
                canvas.setParent(None)
                canvas.deleteLater()

        fig = Figure(figsize=(6, 5))

        if n_selected == 1:
            # 1D line plot
            ax = fig.add_subplot(111)
            x = self.data['data'][var1]
            y = self.data['data']['U'][outvar]

            smp.plotxy(x, y, color='g', ax=ax)
            ax.set_xlabel(var1)
            ax.set_ylabel(outvar)
            
            # Apply log scale if requested
            if hasattr(self, 'log_x_checkbox') and self.log_x_checkbox.isChecked():
                ax.set_xscale('log')
            if hasattr(self, 'log_y_checkbox') and self.log_y_checkbox.isChecked():
                ax.set_yscale('log')
            
            # Ensure plot fills the available space
            ax.set_aspect('auto')

            self.data_canvas_2d = FigureCanvas(fig)
            self.data_plot_stack.addWidget(self.data_canvas_2d)
            self.data_plot_stack.setCurrentWidget(self.data_canvas_2d)
            self.data_canvas_2d.draw()

        elif n_selected == 2:
            x_raw = self.data['data'][var1]
            y_raw = self.data['data'][var2]
            z_raw = self.data['data']['U'][outvar]

            x = x_raw[indices]
            y = y_raw[indices]
            z = z_raw[indices]

            # 2D scatter plot
            fig2d = Figure(figsize=(6, 5))
            ax2d = fig2d.add_subplot(111)                

             
            if self.toggle_mapped_output_in_data_preview.isChecked():
                if np.min(z) < 0.0:
                    print("Plot data with symlog-colors")
                    norm = mcolors.SymLogNorm(vmin=np.min(z), vmax=np.max(z), linthresh=0.01)
                else:
                    print("Plot data with logarithmic colors")
                    # Ensure vmin is positive for LogNorm
                    if np.min(z) == 0.0:
                        min_factor = 1e-16
                        print("Zero's in data to log-scale. Add factor: {}".format(min_factor))
                        vmin = max(np.min(z), min_factor)
                    else:
                        vmin = np.min(z)
                    norm = mcolors.LogNorm(vmin=vmin, vmax=np.max(z))
            else:
                norm = mcolors.Normalize(vmin=np.min(z), vmax=np.max(z))

            sc2d = ax2d.scatter(x, y, c=z, cmap='viridis', norm=norm, s=10)
            print("Min  y: {}".format(np.min(y)))
            ax2d.set_xlabel(var1)
            ax2d.set_ylabel(var2)
            
            # Apply log scale if requested
            if hasattr(self, 'log_x_checkbox') and self.log_x_checkbox.isChecked():
                ax2d.set_xscale('log')
            if hasattr(self, 'log_y_checkbox') and self.log_y_checkbox.isChecked():
                ax2d.set_yscale('log')
                
            # Ensure plot fills the available space
            ax2d.set_aspect('auto')
            ax2d.set_xlim([x.min()-0.05*np.abs((x.max()-x.min())), x.max()+0.05*np.abs((x.max()-x.min()))])
            ax2d.set_ylim([y.min()-0.05*np.abs((y.max()-y.min())), y.max()+0.05*np.abs((y.max()-y.min()))])
            fig2d.colorbar(sc2d, ax=ax2d, label=outvar)

            self.data_canvas_2d = FigureCanvas(fig2d)
            self.data_plot_stack.addWidget(self.data_canvas_2d)
            
            if self.enable_lasso_selector:
                if hasattr(self, "lasso") and self.lasso is not None:
                    self.lasso.disconnect_events()
                    self.lasso = None
                self.lasso = LassoSelector(ax2d, onselect=self.on_lasso_select)
                self.enable_lasso_selector = False

            self.data_canvas_2d.draw()

            # 3D scatter plot
            fig3d = Figure(figsize=(6, 5))
            ax3d = fig3d.add_subplot(111, projection='3d')

            sc3d = ax3d.scatter(x, y, z, c=z, cmap='viridis', s=10)
            ax3d.set_xlabel(var1)
            ax3d.set_ylabel(var2)
            ax3d.set_zlabel(outvar)
            
            # Apply log scale if requested
            if hasattr(self, 'log_x_checkbox') and self.log_x_checkbox.isChecked():
                ax3d.set_xscale('log')
            if hasattr(self, 'log_y_checkbox') and self.log_y_checkbox.isChecked():
                ax3d.set_yscale('log')

            self.data_canvas_3d = FigureCanvas(fig3d)
            self.data_plot_stack.addWidget(self.data_canvas_3d)

            # Show 2D by default
            self.data_plot_stack.setCurrentWidget(self.data_canvas_2d)

        else:
            QtWidgets.QMessageBox.warning(self, "Unsupported",
                                        "More than 2D input not supported.")

    def start_roi_selector(self, ax):
        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            selected_inputs = [w for w in self.input_var_widgets if w[3].isChecked()]
            var1 = selected_inputs[0][0].currentText()
            var2 = selected_inputs[1][0].currentText()
            self.roi = {
                var1: [min(x1, x2), max(x1, x2)],
                var2: [min(y1, y2), max(y1, y2)],
            }
            print("ROI selected:", self.roi)
            self.build_dicts(update_text=True)


        self.rectangle_selector = RectangleSelector(
            ax, onselect,
            useblit=False,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='data',
            interactive=True
        )

        artist = getattr(self.rectangle_selector, '_selection_artist', None)
        if artist:
            artist.set_facecolor('red')
            artist.set_alpha(0.3)
            artist.set_edgecolor('white')

        self.rectangle_selector.set_active(True)
        self.data_canvas_2d.draw()


    def ensure_canvas_2d(self):
        if not hasattr(self, 'data_canvas_2d') or self.data_canvas_2d is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            self.data_canvas_2d = FigureCanvas(fig)
            self.data_ax = ax  # store the axis explicitly
            self.result_plot_stack.addWidget(self.data_canvas_2d)
            self.result_plot_stack.setCurrentWidget(self.data_canvas_2d)


    # New with auto update when element numbers change
    def preview_mesh_overlay(self):
        if not hasattr(self, 'roi') or self.roi is None:
            QtWidgets.QMessageBox.warning(self, "No ROI", "Please draw an ROI before previewing the mesh.")
            return
        try:
            self.ensure_canvas_2d()

            # Get the dimensions selected via the "plot" checkboxes
            selected_inputs = [w for w in self.input_var_widgets if w[3].isChecked()]

            # Determine which dimensions to use (must exist in mesh_specs)
            if len(selected_inputs) >= 2:
                var1 = selected_inputs[0][0].currentText()
                var2 = selected_inputs[1][0].currentText()
                selected_dims = [var1, var2]
            else:
                all_dim_keys = list(self.mesh_specs['element_numbers'].keys())
                selected_dims = all_dim_keys[:2]
                print("Warning: Not enough dimensions selected for plotting. Using first two dimensions.")

            # Filter dimensions to those present in mesh_specs
            valid_dims = [d for d in selected_dims if d in self.mesh_specs.get('element_numbers', {})]
            if len(valid_dims) < 2:
                valid_dims = list(self.mesh_specs.get('element_numbers', {}).keys())[:2]
                print("Warning: Selected dimensions not all present in mesh element_numbers. Using available dimensions:", valid_dims)

            # Build a 2‑D mesh spec copy using only valid dimensions
            mesh_specs_2d = copy.deepcopy(self.mesh_specs)
            mesh_specs_2d['element_numbers'] = {
                k: self.mesh_specs['element_numbers'][k] for k in valid_dims
            }
            if 'mesh_distribution' in mesh_specs_2d:
                mesh_specs_2d['mesh_distribution'] = {
                    k: v for k, v in mesh_specs_2d['mesh_distribution'].items()
                    if k in valid_dims
                }

            # Construct ROI subset matching valid dimensions
            roi_subset = {}
            for dim in valid_dims:
                if isinstance(self.roi, dict) and dim in self.roi:
                    roi_subset[dim] = self.roi[dim]
                else:
                    if dim in self.data['data']:
                        roi_subset[dim] = [float(np.min(self.data['data'][dim])),
                                          float(np.max(self.data['data'][dim]))]
                    else:
                        raise KeyError(f"Cannot determine bounds for dimension '{dim}'")
            # Ensure ROI ranges are non‑zero to avoid Qhull precision issues.
            # If a dimension has an extremely small range (or is flat), expand it slightly.
            epsilon = 1e-8
            for d, bounds in roi_subset.items():
                if abs(bounds[1] - bounds[0]) < epsilon:
                    mid = (bounds[0] + bounds[1]) / 2.0
                    roi_subset[d] = [mid - epsilon/2.0, mid + epsilon/2.0]
                print(f"ROI used for mesh preview: {roi_subset}")

            # Rebuild mesh
            if len(self.mesh_specs['tri_elements']) > 0:
                nodes, conn = fes.build_simplex_extrusion_nd(mesh_specs_2d, roi_subset)
                highlight_elements = fes.data_density_in_element_2d_simplex(
                    self.data['data'], nodes, conn, mesh_specs_2d, verbose=False
                )
            else:
                nodes, conn = fes.build_hypercube_mesh(mesh_specs_2d, roi_subset)
                highlight_elements = fes.data_density_in_element_2d_hypercube(
                    self.data['data'], nodes, conn, mesh_specs_2d, verbose=False,
                    min_points=self.mesh_specs['mesh_filtering']['density_thresh']
                )

            # Store generated mesh
            self.preview_nodes = nodes
            self.preview_conn = conn
            self.mesh_specs['premade_mesh']['nodes'] = nodes.tolist()
            self.mesh_specs['premade_mesh']['conn'] = conn.tolist()
            self.highlight_elements = highlight_elements
            self.mesh_specs['mesh_filtering']['highlight_elements'] = highlight_elements

            # Ensure canvas exists
            if not self.data_canvas_2d:
                self.plot_data()

            # Now overlay on current 2D plot
            ax = self.data_canvas_2d.figure.axes[0]

            # Remove old overlay artists if any
            if hasattr(self, 'mesh_overlay_artists'):
                for artist in self.mesh_overlay_artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass

            # Draw new overlay and keep track of artists
            self.mesh_overlay_artists = smp.visualize_mesh(
                nodes, conn, show_elem_numbers=False, dim=2,
                highlight_elements=highlight_elements,
                show_node_numbers=False, ax=ax,
                return_artists=True
            )

            self.data_canvas_2d.draw()
        except Exception as e:
            print(f"Mesh preview failed: {e}")

        except Exception as e:
            print(f"Mesh preview failed: {e}")


    def plot_time_series(self, DATA=None, UH_fit=None, U_reference=None, outvar=None, input_vars=None):

        print("[plotter] U ref: ", np.min(U_reference) if U_reference is not None else None,
            np.max(U_reference) if U_reference is not None else None)
        print("[plotter] U fit:", np.min(UH_fit) if UH_fit is not None else None,
            np.max(UH_fit) if UH_fit is not None else None)

        if self.data is None and DATA is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        data_source = DATA if DATA is not None else self.data

        # choose x variable
        if input_vars is None:
            selected_inputs = [w for w in self.input_var_widgets if w[3].isChecked()]
            if len(selected_inputs) == 0:
                QtWidgets.QMessageBox.warning(self, "Select X Variable", "Please select the input variable to use as x-axis.")
                return
            input_vars = [w[0].currentText() for w in selected_inputs]

        if len(input_vars) == 1:
            x_var = input_vars[0]
        else:
            x_var, ok = QtWidgets.QInputDialog.getItem(
                self, "Select X Variable",
                "Select input variable for X-axis and segmentation:",
                input_vars, 0, False)
            if not ok or x_var == "":
                return

        # choose output variable
        if outvar is None:
            selected_output = None
            for cb, plot_cb, _, _ in self.output_var_cbs:
                if plot_cb.isChecked():
                    selected_output = cb.currentText()
                    break
            if selected_output is None:
                QtWidgets.QMessageBox.warning(self, "No Output Selected", "Please select an output variable to plot.")
                return
            outvar = selected_output

        # gather arrays
        try:
            x = np.asarray(data_source['data'][x_var])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Missing X", f"Could not find x-variable '{x_var}' in data: {e}")
            return

        y = np.asarray(U_reference) if U_reference is not None else None
        uh = np.asarray(UH_fit) if UH_fit is not None else None

        # Basic length checks
        n = len(x)
        if y is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Reference data could not be determined.")
            return

        if len(y) != n or (uh is not None and len(uh) != n):
            # if lengths differ, try to be forgiving: truncate to min length (but warn)
            lengths = {'x': n, 'y': len(y), 'uh': len(uh) if uh is not None else None}
            min_len = min([v for v in lengths.values() if v is not None])
            print(f"[plotter] Length mismatch detected: {lengths}. Truncating to min length {min_len}.")
            x = x[:min_len]
            y = y[:min_len]
            if uh is not None:
                uh = uh[:min_len]
            n = min_len

        # drop non-finite points consistently (so diff isn't corrupted by NaN/inf)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if uh is not None:
            finite_mask &= np.isfinite(uh)
        if not finite_mask.all():
            print(f"[plotter] Dropping {np.count_nonzero(~finite_mask)} non-finite points.")
            x = x[finite_mask]
            y = y[finite_mask]
            if uh is not None:
                uh = uh[finite_mask]
            n = len(x)

        # Use meta_data segmentation if available and consistent
        meta = None
        if isinstance(data_source, dict) and 'meta_data' in data_source:
            meta = data_source['meta_data']
        elif hasattr(self, 'meta_data') and self.meta_data is not None:
            meta = self.meta_data

        # robust reset detection using decreases in x (with small tolerance)
        rng = np.nanmax(x) - np.nanmin(x)
        abs_tol = max(1e-12, 1e-8 * (rng if rng != 0 else 1.0))

        # If meta is present and consistent, prefer it as authoritative segmentation
        segments = None
        if meta is not None and 'sim_lengths' in meta and sum(meta['sim_lengths']) == len(x):
            # authoritative segmentation
            sim_lengths = np.asarray(meta['sim_lengths'], dtype=int)
            cumsum = np.concatenate(([0], np.cumsum(sim_lengths))).astype(int)
            segments = [np.arange(cumsum[i], cumsum[i+1]) for i in range(len(sim_lengths))]
            print(f"[plotter] Using meta_data segmentation with {len(segments)} segments.")
        else:
            diffs = np.diff(x)
            reset_indices = np.where(diffs < -abs_tol)[0] + 1
            segments = np.split(np.arange(len(x)), reset_indices) if len(reset_indices) > 0 else [np.arange(len(x))]
            print(f"[plotter] Heuristic segmentation found {len(segments)} segments (reset indices: {reset_indices}).")

        # validation & refinement: ensure each segment has monotonically non-decreasing x.
        def refine_segments(segs):
            new_segs = []
            changed = False
            for seg in segs:
                if len(seg) <= 1:
                    new_segs.append(seg)
                    continue
                seg_x = x[seg]
                bad = np.where(np.diff(seg_x) < -abs_tol)[0]  # local decreases
                if bad.size == 0:
                    new_segs.append(seg)
                else:
                    # split at the local decrease points found in this segment
                    splits = (bad + 1).tolist()
                    # global indices where to split relative to seg
                    split_positions = [seg[idx] for idx in splits]
                    # build smaller pieces
                    starts = [seg[0]] + split_positions
                    ends = split_positions + [seg[-1] + 1]
                    for s, e in zip(starts, ends):
                        new_segs.append(np.arange(s, e))
                    changed = True
            return new_segs, changed

        # iteratively refine until no more internal decreases are found (or max iterations)
        max_iters = 10
        for it in range(max_iters):
            segments, changed = refine_segments(segments)
            if not changed:
                break
        if changed:
            print(f"[plotter] Warning: segmentation required refinement after {max_iters} iterations (final segments: {len(segments)}).")

        # now check each segment is monotonic; collect suspicious segments
        suspicious = []
        for j, seg in enumerate(segments):
            if len(seg) > 1:
                if np.any(np.diff(x[seg]) < -abs_tol):
                    suspicious.append(j)

        if len(suspicious) > 0:
            print(f"[plotter] Found {len(suspicious)} suspicious segments (non-monotonic x inside): {suspicious}")
        else:
            print("[plotter] All segments pass monotonicity check.")

        # limit number of plotted lines (random sample if necessary)
        max_lines = 20
        total_segments = len(segments)
        if total_segments > max_lines:
            chosen_idx = np.random.choice(total_segments, size=max_lines, replace=False)
            chosen_idx.sort()
            segments_to_plot = [segments[i] for i in chosen_idx]
            indices_plotted = chosen_idx.tolist()
        else:
            segments_to_plot = segments
            indices_plotted = list(range(len(segments)))

        # plotting
        # clear old canvas and toolbar if present
        if hasattr(self, 'ts_widget') and self.ts_widget:
            self.ts_widget.setParent(None)
            self.ts_widget.deleteLater()
            self.ts_widget = None

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        # colormap
        cmap = cm.get_cmap('tab20', len(segments_to_plot))

        for ic, seg in enumerate(segments_to_plot):
            color = cmap(ic)
            seg_idx_global = indices_plotted[ic]  # which original segment index
            is_suspicious = seg_idx_global in suspicious

            # reference dotted
            ax.plot(x[seg], y[seg], lw=1 if not is_suspicious else 2,
                    linestyle=':', color=('red' if is_suspicious else color),
                    label='Data' if ic == 0 else "")
            # fit solid
            if uh is not None:
                ax.plot(x[seg], uh[seg], lw=1 if not is_suspicious else 2,
                        linestyle='-', color=('darkred' if is_suspicious else color),
                        label='Fit' if ic == 0 else "")

        ax.set_title(f"{outvar} vs {x_var} {'(Data & Fit)' if uh is not None else '(Data)'}")
        ax.set_xlabel(x_var)
        ax.set_ylabel(outvar)
        if uh is not None:
            ax.legend()

        # If suspicious segments exist, notify user and print a brief report
        if len(suspicious) > 0:
            msg = (f"Segmentation suspicious for {len(suspicious)} segments. "
                "Suspicious segments are plotted in red. "
                "Possible causes: non-monotonic x-variable within a simulation, NaNs, "
                "or incorrect concatenation boundaries.")
            print("[plotter] " + msg)
            QtWidgets.QMessageBox.warning(self, "Segmentation Warning", msg)

        # finalize canvas/widget
        self.ts_canvas = FigureCanvas(fig)
        # Add interactive label editing for time series plots
        self.setup_interactive_labels(fig)
        
        self.ts_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ts_canvas)

        self.ts_toolbar = NavigationToolbar(self.ts_canvas, self)
        layout.addWidget(self.ts_toolbar)

        self.ts_widget.setLayout(layout)

        self.result_plot_stack.addWidget(self.ts_widget)
        self.result_plot_stack.setCurrentWidget(self.ts_widget)

        self.ts_canvas.draw()

    def on_time_series_plot_button_clicked(self):
        if not hasattr(self, "DATA2") or not hasattr(self, "outputs"):
            QtWidgets.QMessageBox.warning(self, "No Training Data", "Please run training first.")
            return

        # Determine which output is currently selected
        selected_output = None
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                selected_output = cb.currentText()
                break
        if selected_output is None:
            QtWidgets.QMessageBox.warning(self, "No Output Selected", "Please select an output variable.")
            return

        outvar = selected_output
        arrays = self.outputs[outvar]

        if self.toggle_mapped_output_in_plots.isChecked():
            UH_fit = arrays["uh_mapped"]
            U_reference = arrays["u_mapped"]
        else:
            UH_fit = arrays["uh_physical"]
            U_reference = arrays["u_physical"]

        self.plot_time_series(
            DATA=self.DATA2,
            UH_fit=UH_fit,
            U_reference=U_reference,
            outvar=outvar,
            input_vars=self.input_vars
        )

    def load_training_result(self):
        """Load a previously saved training result (pickle) and update the GUI."""
        # Try to get a stored path first
        file_path = self.args.get("trained_pickle")
        if not file_path or not os.path.isfile(file_path):
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open Trained Result",
                "",
                "Pickle Files (*.pkl *.pickle)"
            )
            if not file_path:
                return  # User cancelled

        try:
            with open(file_path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load the training result file:\\n{e}"
            )
            return

        # -----------------------------------------------------------------
        # Expected pickle structure:
        #   - Individual input arrays (one entry per input variable)
        #   - "U"   : dict of reference output arrays
        #   - "Usm" : dict of surrogate model output arrays
        #   - optional "unmapped" containing original unmapped inputs
        #   - No "train_output" wrapper and no SM in the same file.
        # -----------------------------------------------------------------
        if isinstance(loaded, dict):
            # Extract training outputs
            train_output = {
                "U": loaded.get("U", {}),
                "Usm": loaded.get("Usm", {})
            }

            # Build minimal new_data structure from the remaining entries
            new_data = {"data": {}}
            for key, value in loaded['unmapped'].items():
                if key in ("U", "Usm"):
                    continue
                new_data["data"][key] = value

            # Preserve reference outputs under the "U" key for compatibility
            new_data["data"]["U"] = train_output["U"]

            # Load the surrogate model from the sibling SM.pickle file
            sm_path = os.path.join(os.path.dirname(file_path), "SM.pickle")
            if os.path.isfile(sm_path):
                try:
                    with open(sm_path, "rb") as sm_f:
                        SM = pickle.load(sm_f)
                except Exception as e2:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Load Error",
                        f"Failed to load SM from {sm_path}:\\n{e2}"
                    )
                    return
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load Error",
                    "SM not found – ensure SM.pickle is present in the same folder."
                )
                return
        elif isinstance(loaded, (list, tuple)) and len(loaded) == 2:
            # Legacy format: [train_output, SM]
            train_output, SM = loaded
            new_data = None
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                "The selected file does not contain a valid training result."
            )
            return

        # If new_data could not be constructed (legacy format), attempt a best‑effort reconstruction
        if new_data is None:
            input_vars = list(self.mesh_specs.get("element_numbers", {}).keys())
            output_vars = list(self.args.get("map_output", {}).keys())
            new_data = {"data": {}}
            for var in input_vars:
                if var in SM.get("input_maps", {}):
                    try:
                        inv = SM["input_maps"][var].inverse_transform(
                            SM["mesh"]["nodes"][:, list(self.mesh_specs["element_numbers"].keys()).index(var)]
                        )
                        new_data["data"][var] = np.asarray(inv)
                    except Exception:
                        new_data["data"][var] = np.zeros(SM["mesh"]["nodes"].shape[0])
                else:
                    new_data["data"][var] = np.zeros(SM["mesh"]["nodes"].shape[0])
            for var in output_vars:
                new_data["data"]["U"] = new_data["data"].get("U", {})
                new_data["data"]["U"][var] = train_output["U"].get(var, np.zeros(SM["mesh"]["nodes"].shape[0]))

        # Store loaded results for later use
        self.SM = SM
        self.DATA2 = new_data
        self.input_vars = list(self.mesh_specs.get("element_numbers", {}).keys())
        self.outputs = {}

        for outvar in self.args.get("map_output", {}):
            u_mapped = train_output["U"][outvar]
            uh_mapped = train_output["Usm"][outvar]

            # Convert back to physical space if possible
            try:
                u_physical = SM["output_maps"][outvar].inverse_transform(copy.deepcopy(u_mapped))
                uh_physical = SM["output_maps"][outvar].inverse_transform(copy.deepcopy(uh_mapped))
            except Exception:
                u_physical = SM["output_maps"][outvar].inverse_transform(
                    copy.deepcopy(u_mapped).reshape(-1, 1)
                ).ravel(order="F")
                uh_physical = SM["output_maps"][outvar].inverse_transform(
                    copy.deepcopy(uh_mapped).reshape(-1, 1)
                ).ravel(order="F")

            self.outputs[outvar] = {
                "u_physical": copy.deepcopy(u_physical),
                "u_mapped": copy.deepcopy(u_mapped),
                "uh_mapped": copy.deepcopy(uh_mapped),
                "uh_physical": copy.deepcopy(uh_physical),
            }

        # Determine which output to display initially
        selected_output = None
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                selected_output = cb.currentText()
                break
        if not selected_output:
            # Default to the first available output
            selected_output = list(self.outputs.keys())[0] if self.outputs else None

        if selected_output:
            out_data = self.outputs[selected_output]
            self.update_result_plot(
                SM=self.SM,
                DATA2=self.DATA2,
                u_physical=out_data["u_physical"],
                u_mapped=out_data["u_mapped"],
                uh_mapped=out_data["uh_mapped"],
                uh_physical=out_data["uh_physical"],
                outvar=selected_output,
                input_vars=self.input_vars,
            )
            self._wire_output_selectors()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Load Warning",
                "Training result loaded, but no output variables were found to plot."
            )

        # Remember the file path for future loads
        self.args["trained_pickle"] = file_path

        QtWidgets.QMessageBox.information(self, "Load Successful", f"Training result loaded from:\\n{file_path}")



    def preview_dicts_only(self):
        self.build_dicts(update_text=True)

    def update_dict_preview(self):
        preview = f"mesh_specs:\n{json.dumps(self.mesh_specs, indent=2)}\n\nargs:\n{json.dumps(self.args, indent=2)}"
        if hasattr(self, 'roi') and self.roi:
            preview += f"\n\nroi:\n{json.dumps(self.roi, indent=2)}"
        else:
            preview += "\n\nroi: None"
        self.dict_text.setPlainText(preview)

    def on_lasso_select(self, verts):
        """
        Handle lasso selection for 2D-based refinement.
        Works only on 2D mesh. Higher-dimensional extrusion is deferred.
        """
        from scipy.spatial import Delaunay

        # 1. Refine the selected lasso region
        nodes_sorted, new_nodes = fes.refine_lasso_region(
            preview_nodes=self.preview_nodes,  # should be 2D already
            verts=verts,
            spacing=1e-10,
            decimals=8
        )

        print(f"Generated {len(new_nodes)} new refinement nodes.")

        # 2. Store updated 2D node list
        self.preview_nodes = nodes_sorted

        # 3. Rebuild 2D connectivity (e.g., using Delaunay, as to not extrude into higher dims)
        tri = Delaunay(nodes_sorted)
        conn = tri.simplices
        conn = fes.ensure_positive_orientation_nd(conn, nodes_sorted)
        conn = fes.enforce_checkerboard_diagonals(conn, nodes_sorted)

        self.preview_conn = conn

        # 4. Store in premade_mesh for later extrusion (by training)
        self.mesh_specs.setdefault("premade_mesh", {})
        self.mesh_specs["premade_mesh"]["nodes"] = self.preview_nodes.tolist()
        self.mesh_specs["premade_mesh"]["conn"] = self.preview_conn.tolist()

        # 5. Clean up empty elements once more
        # Make a temporary 2D-only copy of mesh_specs for local use
        mesh_specs_2d = copy.deepcopy(self.mesh_specs)
        mesh_specs_2d['element_numbers'] = {
            k: v for i, (k, v) in enumerate(self.mesh_specs['element_numbers'].items()) if i < 2
        }

        highlight_elements = fes.data_density_in_element_2d_simplex(self.data['data'], nodes_sorted, conn, mesh_specs_2d, verbose=False)
        self.highlight_elements = highlight_elements
        self.mesh_specs['mesh_filtering']['highlight_elements'] = highlight_elements
        
        # 5. Update 2D visualizer
        ax2d = self.data_canvas_2d.figure.get_axes()[0]
        ax2d.clear()

        # === Replot original data points ===
        coord_keys = list(self.mesh_specs['element_numbers'].keys())[:2]
        x = self.data['data'][coord_keys[0]]
        y = self.data['data'][coord_keys[1]]
        selected_output = None
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                selected_output = cb.currentText()
                break

        if not selected_output:
            QtWidgets.QMessageBox.warning(self, "No Output Selected",
                                        "Please select an output variable to plot.")
            return

        u = self.data['data']['U'][selected_output]
        ax2d.scatter(x, y, s=10, c=u, cmap='viridis', alpha=0.4, label='Data')

        smp.visualize_mesh(self.preview_nodes, self.preview_conn,  dim=2, highlight_elements=highlight_elements, show_elem_numbers=False, show_node_numbers=False, ax=ax2d)
        self.data_canvas_2d.draw()

    def build_dicts(self, update_text=False):
        import ast

        input_vars = []
        elements = {}
        tri_elements = []

        for i, (cb, elem_input, tri_cb, plot_cb, scaler_dropdown, scaler_btn) in enumerate(self.input_var_widgets):
            key = cb.currentText()
            if key:
                input_vars.append(key)
                try:
                    val = elem_input.text().strip()
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, int):
                        elements[key] = parsed
                    elif isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                        elements[key] = parsed
                    else:
                        QtWidgets.QMessageBox.warning(self, "Invalid Elements", f"Element input for '{key}' must be an int or list of numbers.")
                        return
                except Exception:
                    QtWidgets.QMessageBox.warning(self, "Invalid Elements", f"Could not parse element input for '{key}'.")
                    return
                if tri_cb.isChecked() and i < 3:
                    tri_elements.append(key)


         # Ensure all input_vars have elements set; assign default if missing
        default_elements = 1
        for key in input_vars:
            if key not in elements:
                elements[key] = default_elements

        output_vars = [cb.currentText() for (cb, _, _, _) in self.output_var_cbs if cb.currentText() in self.output_keys]

        if len(input_vars) == 0 or len(output_vars) == 0:
            print("Not enough input or output variables for training. Abort.")
            return

        # Update or create mesh_distribution for current inputs
        mesh_distribution = {k: {"type": "power", "param": 1.0} for k in input_vars}

        # Preserve existing refinement info if present
        refinement_region = self.mesh_specs.get("refinement_region") if hasattr(self, "mesh_specs") else None
        refinement_configs = self.mesh_specs.get("refinement_configs") if hasattr(self, "mesh_specs") else None

        # Prepare map_input dict based on UI selections, preserve existing config if any
        map_input = {}
        for cb, _, _, _, map_dropdown, scaler_btn in self.input_var_widgets:
            key = cb.currentText()
            if key:
                selected_scaler = map_dropdown.currentText()
                existing_config = self.args.get("map_input", {}).get(key, {}).get(selected_scaler, {}) if hasattr(self, "args") else {}
                map_input[key] = {selected_scaler: existing_config}

        # Set default range for minmax scalers if missing
        for key, scalers in map_input.items():
            if "minmax" in scalers:
                if "range" not in scalers["minmax"] or scalers["minmax"]["range"] is None:
                    scalers["minmax"]["range"] = (-1.0, 1.0)

        # Prepare map_output and constrain_regression in one loop
        map_output = {}
        constrain_regression = {}

        for cb, plot_cb, map_dropdown, bounds_cb in self.output_var_cbs:
            key = cb.currentText()
            if not key:
                continue

            # --- map_output ---
            selected_scaler = map_dropdown.currentText()
            existing_config = (
                self.args.get("map_output", {}).get(key, {}).get(selected_scaler, {})
                if hasattr(self, "args")
                else {}
            )
            map_output[key] = {selected_scaler: existing_config}

            # Ensure minmax has default range
            if "minmax" in map_output[key]:
                if (
                    "range" not in map_output[key]["minmax"]
                    or map_output[key]["minmax"]["range"] is None
                ):
                    map_output[key]["minmax"]["range"] = (-1.0, 1.0)


            # --- constrain_regression ---
            existing_constrain = (
                self.args.get("constrain_regression", {}).get(key)
                if hasattr(self, "args")
                else None
            )

            if bounds_cb.isChecked():
                if isinstance(existing_constrain, dict):
                    # Keep whatever was set by open_bounds_dialog
                    constrain_regression[key] = existing_constrain
                else:
                    constrain_regression[key] = {"lowerbound": None, "upperbound": None}
            else:
                constrain_regression[key] = None

        # === Ensure ROI is set and valid ===
        expected_keys = set(input_vars)
        roi = self.roi if isinstance(self.roi, dict) else {}
        roi_is_missing_keys = not expected_keys.issubset(roi.keys())
        roi_is_bad_format = not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in roi.values())

        if roi_is_missing_keys or roi_is_bad_format:
            self.roi = self.get_roi_from_inputs(input_vars, self.data)


        # (Update dict preview if requested)
        if update_text:
            self.update_dict_preview()

        # Update mesh_specs while preserving existing keys
        self.mesh_specs.update({
            "shapefunc_degree": "linear",
            "element_numbers": elements,
            "mesh_distribution": mesh_distribution,
            "tri_elements": tri_elements,
            "refinement_region": refinement_region,
            "refinement_configs": refinement_configs,
            "mesh_filtering": {"density_thresh": 3, "remove_limit": None}
        })

        # === Initialize premade_mesh placeholder ===
        if "premade_mesh" not in self.mesh_specs:
            self.mesh_specs["premade_mesh"] = {"nodes": None, "conn": None}

        # Preserve existing data_file if present in args
        existing_data_file = self.args.get("data_file") if hasattr(self, "args") else None

        # Update args while preserving keys not overwritten here
        new_args = {
            "output_directory": "./output",
            "map_output": map_output,
            "map_input": map_input,
            "deconcat": False,
            "plot_histograms": False,
            "debug": False,
            "remove_sparse_elements": True,
            "sparse": True,
            "constrain_regression": constrain_regression
        }

        if existing_data_file is not None:
            new_args["data_file"] = existing_data_file

        if hasattr(self, "args"):
            self.args.update(new_args)
        else:
            self.args = new_args

        if update_text:
            self.update_dict_preview()

        # Set the roi 
        expected_keys = set(input_vars)
        roi = self.roi if isinstance(self.roi, dict) else {}
        roi_is_missing_keys = not expected_keys.issubset(roi.keys())
        roi_is_bad_format = not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in roi.values())

        if roi_is_missing_keys or roi_is_bad_format:
            self.roi = self.get_roi_from_inputs(input_vars, self.data)
        
    def show_pca_heatmap_canvas(self):
        """
        Display the PCA heatmap if available. If the canvas has already been created,
        simply switch to it. Otherwise, generate the figure from the cached SM data.
        """
        if self.pca_canvas is not None:
            self.result_plot_stack.setCurrentWidget(self.pca_canvas)
            return

        SM = self._cached_SM
        if SM is None or 'mesh' not in SM or 'M' not in SM['mesh']:
            QtWidgets.QMessageBox.warning(
                self,
                "No PCA Data",
                "No PCA matrix 'M' found in current solution. Please run training first."
            )
            return

        M_dense = np.array(SM['mesh']['M'])

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(M_dense, cmap='hot')
        fig.colorbar(im, ax=ax, shrink=0.7, label="Nodal Influence")

        ax.set_title("Gram Matrix $M = N N^T$")
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")
        ax.set_aspect("auto")

        canvas = FigureCanvas(fig)
        self.pca_canvas = canvas  # cache it
        # Add navigation toolbar for zoom/pan functionality
        if not hasattr(self, 'pca_toolbar') or self.pca_toolbar is None:
            self.pca_toolbar = NavigationToolbar(canvas, self)
            # Insert the toolbar just before adding the canvas to the stack layout
            # Assumes the layout handling is similar to other tabs where toolbar is added to a layout
            # If a specific layout variable exists for PCA, replace accordingly; otherwise, add to the parent layout
            # Find the parent layout (usually the result_tab_layout) and add the toolbar widget
            try:
                self.result_tab_layout.addWidget(self.pca_toolbar)
            except Exception:
                # Fallback: add toolbar directly to the main widget
                self.layout().addWidget(self.pca_toolbar)
        self.result_plot_stack.addWidget(canvas)
        self.result_plot_stack.setCurrentWidget(canvas)
        canvas.draw()


    # TO FIX: mixed up data and results canvases
    def update_result_plot(self, SM=None, DATA2=None, u_physical=None, u_mapped=None,
                       uh_physical=None, uh_mapped=None, outvar=None,
                       input_vars=None, rel_err=None):
        print("Update result plot.")
        # Defensive: avoid incorrect call (e.g., from toggle button passing bool)
        if not isinstance(SM, dict):
            print("Recovered from bad toggle-triggered call.")
            SM = self._cached_SM
            DATA2 = self._cached_DATA2
            u_physical = self._cached_u
            uh_mapped = self._cached_uh
            u_mapped = self._cached_umapped
            uh_physical = self._cached_uhphysical
            outvar = self._cached_outvar
            input_vars = self._cached_input_vars
        else:
            # Save state for toggling later
            self._cached_SM = SM
            self._cached_DATA2 = DATA2
            self._cached_u = u_physical
            self._cached_uh = uh_mapped
            self._cached_umapped = u_mapped
            self._cached_uhphysical = uh_physical
            self._cached_outvar = outvar
            self._cached_input_vars = input_vars

        # Remember which page user was on (only matters for multi-var)
        current_index = self.result_plot_stack.currentIndex() if len(input_vars) > 1 else 0

        # --- Precisely remove ONLY our two canvases if they exist ---
        for attr in ("data_canvas_2d", "data_canvas_3d"):
            canvas = getattr(self, attr, None)
            if canvas is not None:
                idx = self.result_plot_stack.indexOf(canvas)
                if idx != -1:
                    self.result_plot_stack.removeWidget(canvas)
                canvas.setParent(None)
                canvas.deleteLater()
                setattr(self, attr, None)
        # ------------------------------------------------------------

        # Colormap selection
        self.current_cmap = self.colormap_selector.currentText() if hasattr(self, "colormap_selector") else "hot"

        if len(input_vars) == 1:
            # Single variable: one canvas
            fig = Figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            x = DATA2['data'][input_vars[0]]
            nodes = smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps']).flatten()

            # FIX: use the correct outvar map (was 'u' before)
            nodal_values = smd.backtransform_output(
                copy.deepcopy(SM['nodal_values'][outvar]),
                SM['output_maps'][outvar]
            ).flatten()

            smp.plotxy(x, u_physical, color='g', ax=ax)
            smp.plotxy(x, uh_physical, color='r', marker='-', ax=ax)
            smp.plotxy(nodes, nodal_values, color='cyan', marker='o', ax=ax)
            
            # Set up axis labels with the variable names
            ax.set_xlabel(input_vars[0])
            ax.set_ylabel(outvar)
            ax.set_title(f"{outvar} vs {input_vars[0]}")

            self.data_canvas_2d = FigureCanvas(fig)
            self.setup_interactive_labels(fig)
            self.result_plot_stack.addWidget(self.data_canvas_2d)
            self.result_plot_stack.setCurrentIndex(0)
            self.data_canvas_2d.draw()
            return

        # --- Multi-variable: add two canvases ---
        var1, var2 = input_vars[:2]

        # Page 1: 2D Scatter and Mesh
        fig1 = Figure(figsize=(6, 5))
        ax1 = fig1.add_subplot(111)
        fig1.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        if self.toggle_mapped_output_in_plots.isChecked():
            rel_err = 100 * np.abs((u_mapped - uh_mapped) / np.where(u_mapped == 0, np.finfo(float).eps, u_mapped))
        else:
            rel_err = 100 * np.abs((u_physical - uh_physical) / np.where(u_physical == 0, np.finfo(float).eps, u_physical))

        # sc1 = ax1.scatter(
        #     DATA2['data'][var1], DATA2['data'][var2], c=rel_err,
        #     cmap=self.current_cmap, alpha=0.2,
        #     norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), s=10
        # )
        
        triang = tri.Triangulation(DATA2['data'][var1], DATA2['data'][var2])
        
        cf = ax1.tricontourf(triang, rel_err,
                            levels=np.logspace(-1, 2, 50),
                            cmap=self.current_cmap,
                            norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2),
                            extend='both')

        # Overlay original points (helps see where real data is)
        sc1 = ax1.scatter(DATA2['data'][var1], DATA2['data'][var2],
                    c=rel_err, s=10, edgecolor='k', lw=0.4,
                    cmap=self.current_cmap, norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2),
                    zorder=10)
 
        cmap = plt.get_cmap(self.current_cmap)                 # the original cmap
        opaque_cmap = mcolors.ListedColormap(cmap(np.arange(cmap.N)))  # strip alpha channel
        opaque_cmap.set_bad(alpha=1.0)                 # just in case

        cbar = fig1.colorbar(sc1, ax=ax1, cmap=opaque_cmap)
        cbar.solids.set_alpha(1)
        # or, equivalently:
        # cbar = plt.colorbar(sc, ax=ax)
        # cbar.cmap = opaque_cmap   # Matplotlib 3.8+

        cbar.set_label('Relative Error [%]', rotation=270, labelpad=15)

        # fig1.colorbar(sc1, ax=ax1, label='Relative Error [%]', shrink=0.5)
        ax1.set_xlabel(var1)
        ax1.set_ylabel(var2)
        ax1.set_title(f"Relative Error: {outvar}")

        
        # Apply log scale if requested
        if hasattr(self, 'log_x_checkbox') and self.log_x_checkbox.isChecked():
            ax1.set_xscale('log')
        if hasattr(self, 'log_y_checkbox') and self.log_y_checkbox.isChecked():
            ax1.set_yscale('log')
            
        # Ensure plot fills the available space
        ax1.set_aspect('auto')
        ax1.autoscale(enable=True, axis='both', tight=True)

        smp.visualize_mesh(
            smd.backtransform_nodes(copy.deepcopy(SM['mesh']['nodes']), SM['input_maps']),
            SM['mesh']['conn'], dim=2, ax=ax1,
            show_node_numbers=False, show_elem_numbers=False
        )

        self.data_canvas_2d = FigureCanvas(fig1)
        self.setup_interactive_labels(fig1)
        self.result_plot_stack.addWidget(self.data_canvas_2d)

        # Page 2: 3D Surface
        fig2 = Figure(figsize=(7, 5))
        ax2 = fig2.add_subplot(111, projection='3d')
        fig2.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        max_points = self.num_points_spinbox.value()
        num_points = len(DATA2['data'][var1])
        indices = np.random.choice(num_points, min(max_points, num_points), replace=False)

        x1_ds = DATA2['data'][var1][indices]
        x2_ds = DATA2['data'][var2][indices]

        if self.toggle_mapped_output_in_plots.isChecked():
            u_ds = u_mapped[indices]
            uh_ds = uh_mapped[indices]
        else:
            u_ds = u_physical[indices]
            uh_ds = uh_physical[indices]

        if self.toggle_data_points.isChecked():
            ax2.scatter(x1_ds, x2_ds, u_ds, marker='o', color='r', s=40, alpha=0.5, edgecolors='k')
            ax2.scatter(x1_ds, x2_ds, uh_ds, color='k', marker='s', s=50, alpha=0.5, edgecolor=(0, 0, 0, 0))

        if self.toggle_trisurf.isChecked():
            ax2.plot_trisurf(x1_ds, x2_ds, uh_ds, cmap='viridis', alpha=0.4, edgecolors='k')

        if self.toggle_nodes.isChecked():
            
            # Backtransform nodal coordinates to physical space
            nodes_xy = smd.backtransform_nodes(
                copy.deepcopy(SM['mesh']['nodes']),
                SM['input_maps']
            )

            # Backtransform nodal values to physical space (or not)
            if self.toggle_mapped_output_in_plots.isChecked() == False:
                nodal_values = smd.backtransform_output(
                    copy.deepcopy(SM['nodal_values'][outvar]),
                    SM['output_maps'][outvar]
                ).flatten()
            elif self.toggle_mapped_output_in_plots.isChecked():
                nodal_values = copy.deepcopy(SM['nodal_values'][outvar])

            # Assume 2 input dims, so take first 2 cols as x,y
            x_nodes = nodes_xy[:, 0]
            y_nodes = nodes_xy[:, 1]
            z_nodes = nodal_values

            # Scatter plot of nodes
            ax2.scatter(
                x_nodes, y_nodes, z_nodes,
                marker='^', color='yellow', s=50,
                edgecolors='green', alpha=0.8, label='Nodal values'
            )


        ax2.legend(loc="best")
        ax2.set_xlabel(var1)
        ax2.set_ylabel(var2)
        ax2.set_zlabel(outvar)
        ax2.set_title(f"3D Surface Plot: {outvar}")
            
        # Ensure plot fills the available space
        ax2.set_aspect('auto')
        ax2.autoscale(enable=True, axis='both', tight=True)

        self.data_canvas_3d = FigureCanvas(fig2)
        self.setup_interactive_labels(fig2)
        self.result_plot_stack.addWidget(self.data_canvas_3d)

        # Restore page user was on (or default to first)
        target_idx = min(current_index, self.result_plot_stack.count() - 1)
        self.result_plot_stack.setCurrentIndex(target_idx)

        self.data_canvas_2d.draw()
        self.data_canvas_3d.draw()

    def _wire_output_selectors(self):
        # Create/clear an exclusive group for the plot checkboxes
        if not hasattr(self, "_plot_group"):
            self._plot_group = QtWidgets.QButtonGroup(self)
            self._plot_group.setExclusive(True)

        # Remove previously added buttons to avoid duplicates across retrains
        for b in list(self._plot_group.buttons()):
            self._plot_group.removeButton(b)

        for cb, plot_cb, _, _ in self.output_var_cbs:
            self._plot_group.addButton(plot_cb)

            # Disconnect old connections to avoid multiple triggers
            try: cb.currentIndexChanged.disconnect()
            except TypeError: pass
            try: plot_cb.toggled.disconnect()
            except TypeError: pass

            # If this checkbox gets turned ON, replot with its combo's current text
            plot_cb.toggled.connect(lambda checked, combo=cb: checked and self.plot_output(combo.currentText()))

            # If the combo changes, replot only if THIS checkbox is the active one
            cb.currentIndexChanged.connect(lambda _, combo=cb, chk=plot_cb: chk.isChecked() and self.plot_output(combo.currentText()))

        # Enforce exactly one checked (if none, check the first; if many, leave only the first)
        checked = [b for b in self._plot_group.buttons() if b.isChecked()]
        if len(checked) == 0 and self._plot_group.buttons():
            self._plot_group.buttons()[0].setChecked(True)
        elif len(checked) > 1:
            # keep only the first checked
            for b in checked[1:]:
                b.setChecked(False)

    # NEW helper for output plotting (switching outputs)
    def plot_output(self, outvar: str):
        if not hasattr(self, "outputs") or outvar not in self.outputs:
            print(f"Output {outvar} not available.")
            return
        outputs = self.outputs[outvar]
        self.update_result_plot(
            SM=self.SM,
            DATA2=self.DATA2,
            u_physical=outputs["u_physical"],
            u_mapped=outputs["u_mapped"],
            uh_mapped=outputs["uh_mapped"],
            uh_physical=outputs["uh_physical"],
            outvar=outvar,
            input_vars=self.input_vars,
        )


    def get_roi_from_inputs(self, input_vars, data):
        roi = {}
        for k in input_vars:
            if self.roi and k in self.roi:
                roi[k] = self.roi[k]
            else:
                # TOFIX: add a little margin to make sure all data is inside the ROI
                roi[k] = [min(data['data'][k]), max(data['data'][k])]
        return roi

    def run_training(self):
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        self.build_dicts(update_text=True)

        input_vars = list(self.mesh_specs["element_numbers"].keys())
        output_vars = list(self.args["map_output"].keys())
        print("inputs: {}".format(input_vars))

        if len(input_vars) == 0 or len(output_vars) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient variables",
                "Select at least one input and one output for training."
            )
            return

        # Determine which output is selected for initial plotting
        selected_output = None
        for cb, plot_cb, _, _ in self.output_var_cbs:
            if plot_cb.isChecked():
                selected_output = cb.currentText()
                break

        if not selected_output:
            QtWidgets.QMessageBox.warning(
                self,
                "Output Selection",
                "Please select an output variable to plot."
            )
            return

        new_data = {
            "data": {
                "x": np.vstack([self.data["data"][k] for k in input_vars]).T,
                "U": {k: self.data["data"]["U"][k] for k in output_vars},
            }
        }
        for k in input_vars:
            new_data["data"][k] = self.data["data"][k]

        roi = self.roi
        print("ROI in training: {}".format(roi))

        # Add data_file to args if not already present
        if "data_file" in self.args and self.args["data_file"]:
            # Store data_file for future reference
            data_file_path = self.args["data_file"]
            new_data["data_file"] = data_file_path
        
        
        train_output, SM = smb.sm_train(
            roi, self.mesh_specs, self.args, DATA=copy.deepcopy(new_data)
        )
        
        # Store data_file path in SM dictionary
        if "data_file" in self.args and self.args["data_file"]:
            SM["data_file"] = self.args["data_file"]
        
        # Store data_file path in SM dictionary
        if "data_file" in self.args and self.args["data_file"]:
            SM["data_file"] = self.args["data_file"]

        # analyze aspect ratio
        if len(input_vars) > 1:
            bad_elements = sma.analyze_prism_quality(
                SM["mesh"]["nodes"], SM["mesh"]["conn"],
                self.mesh_specs, show_plot=True
            )

        # Store results for *all* outputs
        print("Store outputs for plotting.")
        self.outputs = {}  # reset on each run
        for outvar in output_vars:
            u_mapped = train_output["U"][outvar]
            uh_mapped = train_output["Usm"][outvar]
            try:
                u_physical = SM["output_maps"][outvar].inverse_transform(copy.deepcopy(u_mapped))
                uh_physical = SM["output_maps"][outvar].inverse_transform(copy.deepcopy(uh_mapped))
            except ValueError:
                u_physical = SM["output_maps"][outvar].inverse_transform(copy.deepcopy(u_mapped).reshape(-1, 1)).ravel(order="F")
                uh_physical = SM["output_maps"][outvar].inverse_transform(copy.deepcopy(uh_mapped).reshape(-1, 1)).ravel(order="F")

            self.outputs[outvar] = {
                "u_physical": copy.deepcopy(u_physical),
                "u_mapped": copy.deepcopy(u_mapped),
                "uh_mapped": copy.deepcopy(uh_mapped),
                "uh_physical": copy.deepcopy(uh_physical),
            }

        # Plot the initially selected output
        outputs = self.outputs[selected_output]
        self.update_result_plot(
            SM=SM,
            DATA2=new_data,
            u_physical=outputs["u_physical"],
            u_mapped=outputs["u_mapped"],
            uh_mapped=outputs["uh_mapped"],
            uh_physical=outputs["uh_physical"],
            outvar=selected_output,
            input_vars=input_vars,
        )

        # Save shared data for later
        self.DATA2 = new_data
        self.SM = SM
        self.input_vars = input_vars

        # Wire checkboxes and combo boxes for post-training replotting
        self._wire_output_selectors()

        self.save_state()
    
    def save_merged_data(self):
        """Prompt for a location and save the currently merged dataset to a pickle file."""
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "There is no merged data to save.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Merged Data",
            "",
            "Pickle Files (*.pkl *.pickle)"
        )
        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.data, f)
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Merged data saved successfully to:\n{file_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to save merged data:\n{e}"
            )

class BoundsSettingsDialog(QtWidgets.QDialog):
    def __init__(self, var_name, lower=None, upper=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Bounds for {var_name}")
        self.setModal(True)

        layout = QtWidgets.QFormLayout()

        # --- Lower bound ---
        self.lower_combo = QtWidgets.QComboBox()
        self.lower_combo.addItems(["None", "Use Min", "Custom"])
        self.lower_edit = QtWidgets.QLineEdit()
        self.lower_edit.setPlaceholderText("Enter number")
        self.lower_edit.setEnabled(False)  # only enabled if "Custom" is selected

        layout.addRow("Lower bound:", self.lower_combo)
        layout.addRow("", self.lower_edit)

        # --- Upper bound ---
        self.upper_combo = QtWidgets.QComboBox()
        self.upper_combo.addItems(["None", "Use Max", "Custom"])
        self.upper_edit = QtWidgets.QLineEdit()
        self.upper_edit.setPlaceholderText("Enter number")
        self.upper_edit.setEnabled(False)

        layout.addRow("Upper bound:", self.upper_combo)
        layout.addRow("", self.upper_edit)

        # --- Signals ---
        self.lower_combo.currentIndexChanged.connect(lambda idx: self.lower_edit.setEnabled(idx == 2))
        self.upper_combo.currentIndexChanged.connect(lambda idx: self.upper_edit.setEnabled(idx == 2))

        # --- Buttons ---
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

        # --- Set initial values ---
        self.set_initial("lower", lower)
        self.set_initial("upper", upper)

    def set_initial(self, which, value):
        combo = self.lower_combo if which == "lower" else self.upper_combo
        edit = self.lower_edit if which == "lower" else self.upper_edit

        if value is None:
            combo.setCurrentText("None")
        elif value == "min":
            combo.setCurrentText("Use Min")
        elif value == "max":
            combo.setCurrentText("Use Max")
        else:
            combo.setCurrentText("Custom")
            edit.setText(str(value))
            edit.setEnabled(True)

    def get_bounds(self):
        """Return a tuple (lower, upper), each can be None, 'min', 'max', or float"""
        def parse(combo, edit):
            choice = combo.currentText()
            if choice == "None":
                return None
            elif choice == "Use Min":
                return "min"
            elif choice == "Use Max":
                return "max"
            else:  # Custom
                try:
                    return float(edit.text())
                except ValueError:
                    return None

        lower = parse(self.lower_combo, self.lower_edit)
        upper = parse(self.upper_combo, self.upper_edit)
        return lower, upper


class ScalerSettingsDialog(QtWidgets.QDialog):
        def __init__(self, var_name, scaler_type, config=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle(f"Configure Scaler for {var_name} [{scaler_type}]")
            self.scaler_type = scaler_type
            self.config = config or {}

            self.layout = QtWidgets.QFormLayout()
            self.setLayout(self.layout)

            # Add fields based on scaler_type
            self.fields = {}


            if scaler_type == "minmax":
                self.fields["range"] = self.add_range_field("range", self.config.get("range", (-1, 1)))
            elif scaler_type == "log10":
                self.fields["factor"] = self.add_line_field("factor", self.config.get("factor", None))
                self.fields["lowerbound"] = self.add_line_field("lowerbound", self.config.get("lowerbound", 0.0))
                self.fields["upperbound"] = self.add_line_field("upperbound", self.config.get("upperbound", 1.0))
            elif scaler_type == "symlog":
                self.fields["lowerbound"] = self.add_line_field("lowerbound", self.config.get("lowerbound", 0.0))
                self.fields["upperbound"] = self.add_line_field("upperbound", self.config.get("upperbound", 1.0))
            elif scaler_type == "compress":
                self.fields["factor"] = self.add_line_field("factor", self.config.get("factor", 1e-10))
                self.fields["compressor"] = self.add_line_field("compressor", self.config.get("compressor", 0.01))

            # OK/Cancel buttons
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            self.layout.addRow(buttons)

        def add_line_field(self, label, default):
            display_text = "" if default is None else str(default)
            line = QtWidgets.QLineEdit(display_text)
            self.layout.addRow(label, line)
            return line

        def add_range_field(self, label, default):
            container = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)
            min_edit = QtWidgets.QLineEdit(str(default[0]))
            max_edit = QtWidgets.QLineEdit(str(default[1]))
            hbox.addWidget(QtWidgets.QLabel("Min:"))
            hbox.addWidget(min_edit)
            hbox.addWidget(QtWidgets.QLabel("Max:"))
            hbox.addWidget(max_edit)
            container.setLayout(hbox)
            self.layout.addRow(label, container)
            return (min_edit, max_edit)

        def get_config(self):
            if self.scaler_type == "minmax":
                min_val = float(self.fields["range"][0].text())
                max_val = float(self.fields["range"][1].text())
                return {"range": (min_val, max_val)}
            else:
                out = {}
                for k, widget in self.fields.items():
                    try:
                        out[k] = float(widget.text()) if widget.text() else None
                    except Exception:
                        out[k] = widget.text()
                return out

class MeshSettingsDialog(QtWidgets.QDialog):
        def __init__(self, parent, input_keys):
            super().__init__(parent)
            self.setWindowTitle("Refinement Settings")
            self.setModal(True)
            self.input_keys = input_keys
            self.setup_ui()

        def _hbox(self, *widgets):
            box = QtWidgets.QHBoxLayout()
            for w in widgets:
                box.addWidget(w)
            container = QtWidgets.QWidget()
            container.setLayout(box)
            return container

        def get_refinement_settings(self):
            if not self.enable_checkbox.isChecked():
                return None, None
            var1 = self.var1_cb.currentText()
            var2 = self.var2_cb.currentText()

            region = {
                var1: (float(self.min1.text()), float(self.max1.text())),
                var2: (float(self.min2.text()), float(self.max2.text()))
            }
            configs = {
                "spacing": float(self.spacing_box.value()),
                "mode": self.mode_box.currentText()
            }

            return region, configs
        
        def setup_ui(self):
            layout = QtWidgets.QFormLayout()

            self.enable_checkbox = QtWidgets.QCheckBox("Enable refinement")
            layout.addRow(self.enable_checkbox)

            self.var1_cb = QtWidgets.QComboBox()
            self.var2_cb = QtWidgets.QComboBox()
            self.var1_cb.addItems(self.input_keys)
            self.var2_cb.addItems(self.input_keys)
            layout.addRow("Refine dimension 1:", self.var1_cb)
            layout.addRow("Refine dimension 2:", self.var2_cb)

            self.min1 = QtWidgets.QLineEdit("0")
            self.max1 = QtWidgets.QLineEdit("1")
            self.min2 = QtWidgets.QLineEdit("0")
            self.max2 = QtWidgets.QLineEdit("1")
            layout.addRow("Bounds var1 (min, max):", self._hbox(self.min1, self.max1))
            layout.addRow("Bounds var2 (min, max):", self._hbox(self.min2, self.max2))

            self.spacing_box = QtWidgets.QDoubleSpinBox()
            self.spacing_box.setDecimals(2)  # You can adjust precision as needed
            self.spacing_box.setRange(0.01, 50.0)
            self.spacing_box.setSingleStep(0.1)

            self.mode_box = QtWidgets.QComboBox()
            self.mode_box.addItems(["grid", "random"])
            layout.addRow("Spacing:", self.spacing_box)
            layout.addRow("Mode:", self.mode_box)

            self.ok_btn = QtWidgets.QPushButton("OK")
            self.ok_btn.clicked.connect(self.accept)
            self.cancel_btn = QtWidgets.QPushButton("Cancel")
            self.cancel_btn.clicked.connect(self.reject)

            btn_layout = QtWidgets.QHBoxLayout()
            btn_layout.addWidget(self.ok_btn)
            btn_layout.addWidget(self.cancel_btn)
            layout.addRow(btn_layout)

            self.setLayout(layout)

        # -------------------------------------------------------------------------
        # New functionality: load additional datasets and consolidate columns
        # -------------------------------------------------------------------------
        def add_more_data(self):
            """Open a file dialog to load an additional dataset and merge it with the
            currently loaded data, handling missing columns by augmenting with a user‑
            supplied fill value (default 0.0)."""
            # Prompt user for the additional file
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Additional Pickle File", "", "Pickle Files (*.pkl *.pickle)"
            )
            if not file_path:
                return

            # Load the new dataset
            try:
                with open(file_path, 'rb') as f:
                    new_data = pickle.load(f)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Error",
                                            f"Failed to load the selected file:\\n{e}")
                return

            # Determine which columns are missing in either dataset
            existing_inputs = set(k for k in self.data['data'].keys() if k not in ('U', 'x'))
            new_inputs = set(k for k in new_data['data'].keys() if k not in ('U', 'x'))

            existing_outputs = set(self.data['data']['U'].keys())
            new_outputs = set(new_data['data']['U'].keys())

            missing_in_existing = new_inputs.union(new_outputs) - existing_inputs.union(existing_outputs)
            missing_in_new = existing_inputs.union(existing_outputs) - new_inputs.union(new_outputs)

            # If there are missing columns, ask the user for a fill value
            fill_value = 0.0
            if missing_in_existing or missing_in_new:
                missing_info = []
                if missing_in_existing:
                    missing_info.append("Columns missing in the currently loaded data:")
                    for col in sorted(missing_in_existing):
                        missing_info.append(f"  - {col}")
                if missing_in_new:
                    missing_info.append("Columns missing in the newly loaded data:")
                    for col in sorted(missing_in_new):
                        missing_info.append(f"  - {col}")

                dlg = self.MissingColumnsDialog("\n".join(missing_info), parent=self)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    try:
                        fill_value = float(dlg.get_fill_value())
                    except ValueError:
                        QtWidgets.QMessageBox.warning(self, "Invalid Fill",
                                                    "Fill value must be a number. Using default 0.0.")
                        fill_value = 0.0
                else:
                    # User cancelled the merge
                    return

            # Perform the consolidation/merge
            self._consolidate_data(new_data, fill_value)

            # Update the file‑path label to reflect the combined files
            current_text = self.file_path_label.text()
            combined_text = f"{current_text} + {self.truncate_path(file_path, max_length=100)}"
            self.file_path_label.setText(combined_text)

        def _consolidate_data(self, new_data, fill_value=0.0):
            """Merge `new_data` into the current `self.data` ensuring both have the same columns."""
            # Helper to create a column filled with `fill_value`
            def filler(length):
                return np.full(length, fill_value, dtype=float)

            # Existing lengths
            existing_len = len(next(iter(self.data['data'].values()))) if self.data['data'] else 0
            new_len = len(next(iter(new_data['data'].values()))) if new_data['data'] else 0

            # Union of input columns (excluding 'U')
            all_input_cols = set(k for k in self.data['data'].keys() if k not in ('U', 'x')).union(
                            set(k for k in new_data['data'].keys() if k not in ('U', 'x')))

            # Union of output columns
            all_output_cols = set(self.data['data']['U'].keys()).union(set(new_data['data']['U'].keys()))

            # Build merged data dict
            merged = {'data': {}}

            # Process input columns
            for col in all_input_cols:
                if col in self.data['data']:
                    arr_existing = np.asarray(self.data['data'][col])
                else:
                    arr_existing = filler(existing_len)

                if col in new_data['data']:
                    arr_new = np.asarray(new_data['data'][col])
                else:
                    arr_new = filler(new_len)

                merged['data'][col] = np.concatenate([arr_existing, arr_new])

            # Process output columns under the 'U' key
            merged['data']['U'] = {}
            for col in all_output_cols:
                if col in self.data['data']['U']:
                    arr_existing = np.asarray(self.data['data']['U'][col])
                else:
                    arr_existing = filler(existing_len)

                if col in new_data['data']['U']:
                    arr_new = np.asarray(new_data['data']['U'][col])
                else:
                    arr_new = filler(new_len)

                merged['data']['U'][col] = np.concatenate([arr_existing, arr_new])

            # Replace current data with merged result
            self.data = merged

            # Refresh internal key lists
            self.input_keys = [k for k in self.data['data'].keys() if k not in ('U', 'x')]
            self.output_keys = list(self.data['data']['U'].keys())

            # Refresh combo boxes in the UI (preserve current selections where possible)
            for cb, *_ in self.input_var_widgets:
                current = cb.currentText()
                cb.clear()
                cb.addItems(self.input_keys)
                if current in self.input_keys:
                    cb.setCurrentText(current)

            for cb, *_ in self.output_var_cbs:
                current = cb.currentText()
                cb.clear()
                cb.addItems(self.output_keys)
                if current in self.output_keys:
                    cb.setCurrentText(current)

        # -------------------------------------------------------------------------
        # Helper dialog to show missing columns and ask for a fill value
        # -------------------------------------------------------------------------
        class MissingColumnsDialog(QtWidgets.QDialog):
            def __init__(self, missing_text, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Missing Columns Detected")
                self.setModal(True)

                layout = QtWidgets.QVBoxLayout()

                # Read‑only text area showing missing column information
                info = QtWidgets.QTextEdit()
                info.setReadOnly(True)
                info.setPlainText(missing_text)
                layout.addWidget(info)

                # Fill value entry
                fill_layout = QtWidgets.QHBoxLayout()
                fill_layout.addWidget(QtWidgets.QLabel("Fill value for missing columns:"))
                self.fill_edit = QtWidgets.QLineEdit("0.0")
                fill_layout.addWidget(self.fill_edit)
                layout.addLayout(fill_layout)

                # OK / Cancel buttons
                buttons = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                )
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addWidget(buttons)

                self.setLayout(layout)

            def get_fill_value(self):
                return self.fill_edit.text()
            

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SurrogateModelApp()
    window.show()
    sys.exit(app.exec_())
