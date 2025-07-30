import sys
import numpy as np
import tcod
import google.generativeai as genai
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import os
import random
import json
import logging
import ui_styles

logging.basicConfig(filename='dm_assistant.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming these are in your project structure
from database import setup_database, seed_data_if_needed, SessionLocal, Monster, MagicItem, Armor, Weapon, Spell, NPC
from workers import (GeminiWorker, AudioTranscriberWorker, MonsterImporterWorker,
                     MagicItemImporterWorker, ArmorImporterWorker, WeaponImporterWorker, SpellImporterWorker)
from ui_dialogs import (ResponseDialog, AddMonsterDialog, MonsterDetailDialog,
                        FilterMonsterDialog, MagicItemDetailDialog, FilterMagicItemDialog,
                        ArmorDetailDialog, FilterArmorDialog, WeaponDetailDialog, FilterWeaponDialog,
                        SpellDetailDialog, FilterSpellDialog, AddEditNPCDialog)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QTabWidget, QSplitter, QGraphicsView,
                             QGraphicsScene, QGraphicsRectItem, QToolBar,
                             QStatusBar, QMessageBox, QDialog, QFileDialog,
                             QComboBox, QTextBrowser, QListWidget, QListWidgetItem,
                             QProgressBar, QSlider, QGroupBox, QFrame, QInputDialog,
                             QGraphicsSimpleTextItem, QAbstractItemView, QCheckBox)
from PyQt6.QtGui import QAction, QColor, QBrush, QPen, QIntValidator, QImage, QPainter, QFont, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QRectF

class DungeonMasterAssistant(QMainWindow):
    LAST_STATE_FILE = "last_combat_state_path.txt"
    CONFIG_FILE = "config.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dungeon Master Assistant")
        self.setGeometry(100, 100, 1200, 800)

        setup_database()
        seed_data_if_needed()

        self.rng = tcod.random.Random()
        self.gemini_worker = None
        self.transcriber_worker = None
        self.monster_importer_worker = None
        self.magic_item_importer_worker = None
        self.armor_importer_worker = None
        self.weapon_importer_worker = None
        self.spell_importer_worker = None

        self.recording_state = "stopped"
        self.recorded_frames = []
        self.audio_stream = None
        self.sample_rate = 44100
        self.elapsed_seconds = 0
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self._update_timer_display)
        self.level_check_timer = QTimer(self)
        self.level_check_timer.timeout.connect(self._update_mic_level)
        self.last_peak_level = 0.0

        self.playback_stream = None
        self.loaded_audio_data = None
        self.current_playback_filepath = None
        self.current_frame = 0
        self.session_timestamps = []

        self.current_map_rooms = []
        self.placed_entity_items = []
        self.active_map_entities = []
        self.current_entity_filters = {
            "Monsters": {},
            "Magic Items": {},
            "Armor": {},
            "Weapons": {}
        }

        self.gemini_model = None

        self.central_widget = QTabWidget(self)
        self.setCentralWidget(self.central_widget)

        # Initialize combat state
        self.ailments = {}  # Dictionary for ailment tracking
        self.tab_initialized = {}  # Dictionary for tab initialization
        self.current_turn_row = -1
        self.current_round = 1
        self.first_combatant_name_for_round_check = ""

        # Create tabs
        self.create_dashboard_tab()
        self.create_entity_management_tab()
        self.create_combat_tracker_tab()
        self.create_map_generator_tab()
        self.create_session_manager_tab()
        self.create_ai_assistant_tab()

        self._create_toolbar()
        self._create_statusbar()

        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._auto_save_combat_state)
        self.auto_save_interval = 120 * 1000
        self.auto_save_timer.start(self.auto_save_interval)
        self.current_combat_file_path = None

        last_path = self._load_last_state_path()
        if last_path:
            self.statusBar().showMessage(
                f"Attempting to auto-load last combat state from: {os.path.basename(last_path)}")
            QTimer.singleShot(100, lambda: self._load_combat_state(file_path=last_path))
            self.current_combat_file_path = last_path
        else:
            self.statusBar().showMessage("Ready (No previous combat state to auto-load)")
            self.current_combat_file_path = "auto_save_combat_state.json"
            self._save_last_state_path(self.current_combat_file_path)

        QTimer.singleShot(50, self._get_gemini_api_key)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        action_new_campaign = QAction("New Campaign", self)
        action_save_campaign = QAction("Save Campaign", self)
        action_exit = QAction("Exit", self)
        action_exit.triggered.connect(self.close)
        toolbar.addAction(action_new_campaign)
        toolbar.addAction(action_save_campaign)
        toolbar.addSeparator()
        toolbar.addAction(action_exit)

    def _create_statusbar(self):
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    def create_dashboard_tab(self):
        dashboard_tab = QWidget(self)
        layout = QVBoxLayout(dashboard_tab)
        layout.addWidget(QLabel("<h1>Campaign Dashboard</h1>", self))
        layout.addWidget(QLabel("Welcome, Dungeon Master!", self))
        self.central_widget.addTab(dashboard_tab, "Dashboard")

    def create_entity_management_tab(self):
        self.entity_tab_widget = QWidget()
        layout = QVBoxLayout(self.entity_tab_widget)
        self.entity_sub_tabs = QTabWidget()
        layout.addWidget(self.entity_sub_tabs)

        # Create placeholder widgets for each tab
        self.monster_tab_content = QWidget()
        self.item_tab_content = QWidget()
        self.armor_tab_content = QWidget()
        self.weapon_tab_content = QWidget()
        self.spell_tab_content = QWidget()
        self.npc_tab_content = QWidget()  # New placeholder for NPCs

        self.entity_sub_tabs.addTab(self.monster_tab_content, "Monsters")
        self.entity_sub_tabs.addTab(self.item_tab_content, "Magic Items")
        self.entity_sub_tabs.addTab(self.armor_tab_content, "Armor")
        self.entity_sub_tabs.addTab(self.weapon_tab_content, "Weapons")
        self.entity_sub_tabs.addTab(self.spell_tab_content, "Spells")
        self.entity_sub_tabs.addTab(self.npc_tab_content, "NPCs")  # Add the tab

        # Refresh all tabs to populate them
        self._refresh_monster_tab()
        self._refresh_item_tab()
        self._refresh_armor_tab()
        self._refresh_weapon_tab()
        self._refresh_spell_tab()
        self._refresh_npc_tab()  # New refresh call for NPCs

        self.central_widget.addTab(self.entity_tab_widget, "Entity Management")

    def _replace_entity_tab(self, name, widget, index):
        current_text = ""
        if self.entity_sub_tabs.currentIndex() != -1:
            current_text = self.entity_sub_tabs.tabText(self.entity_sub_tabs.currentIndex())

        self.entity_sub_tabs.removeTab(index)
        self.entity_sub_tabs.insertTab(index, widget, name)

        if current_text == name:
            self.entity_sub_tabs.setCurrentIndex(index)

    def _refresh_monster_tab(self):
        db_session = SessionLocal()
        results = db_session.query(Monster).order_by(Monster.name).all()
        db_session.close()
        self.monster_tab_content = self.create_entity_sub_tab("Monsters", ["Name", "CR", "HP", "AC"])
        self._populate_monster_table(results)
        self._replace_entity_tab("Monsters", self.monster_tab_content, 0)

    def _refresh_item_tab(self):
        db_session = SessionLocal()
        results = db_session.query(MagicItem).order_by(MagicItem.name).all()
        db_session.close()
        self.item_tab_content = self.create_entity_sub_tab("Magic Items", ["Name", "Type", "Rarity"])
        self._populate_item_table(results)
        self._replace_entity_tab("Magic Items", self.item_tab_content, 1)

    def _refresh_armor_tab(self):
        db_session = SessionLocal()
        results = db_session.query(Armor).order_by(Armor.name).all()
        db_session.close()
        self.armor_tab_content = self.create_entity_sub_tab("Armor", ["Name", "Category", "AC"])
        self._populate_armor_table(results)
        self._replace_entity_tab("Armor", self.armor_tab_content, 2)

    def _refresh_weapon_tab(self):
        db_session = SessionLocal()
        results = db_session.query(Weapon).order_by(Weapon.name).all()
        db_session.close()
        self.weapon_tab_content = self.create_entity_sub_tab("Weapons", ["Name", "Category", "Damage"])
        self._populate_weapon_table(results)
        self._replace_entity_tab("Weapons", self.weapon_tab_content, 3)

    def _refresh_spell_tab(self):
        db_session = SessionLocal()
        results = db_session.query(Spell).order_by(Spell.level_int, Spell.name).all()
        db_session.close()
        self.spell_tab_content = self.create_entity_sub_tab("Spells", ["Name", "Level", "School"])
        self._populate_spell_table(results)
        self._replace_entity_tab("Spells", self.spell_tab_content, 4)

    def create_entity_sub_tab(self, name, headers, data=None):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        if name == "Monsters":
            self.monster_table = table
            table.itemDoubleClicked.connect(self._show_monster_details)
            api_group = QGroupBox("Monster Database")
            api_layout = QHBoxLayout()
            self.monster_search_input = QLineEdit()
            self.monster_search_input.setPlaceholderText("Search local monsters by name...")
            search_button = QPushButton("Search")
            search_button.clicked.connect(self._on_local_search_clicked)
            filter_button = QPushButton("Advanced Filter...")
            filter_button.clicked.connect(self._open_filter_dialog)
            import_button = QPushButton("Import All SRD Monsters")
            import_button.setObjectName("import_all_button")
            import_button.clicked.connect(self._on_import_all_monsters_clicked)
            api_layout.addWidget(self.monster_search_input)
            api_layout.addWidget(search_button)
            api_layout.addWidget(filter_button)
            api_layout.addWidget(import_button)
            api_layout.addStretch()
            api_group.setLayout(api_layout)
            layout.addWidget(api_group)
        elif name == "Magic Items":
            self.item_table = table
            table.itemDoubleClicked.connect(self._show_magic_item_details)
            controls_group = QGroupBox("Magic Item Database")
            controls_layout = QHBoxLayout()
            filter_button = QPushButton("Advanced Filter...")
            filter_button.clicked.connect(self._open_item_filter_dialog)
            import_button = QPushButton("Import All SRD Magic Items")
            import_button.setObjectName("import_all_button")
            import_button.clicked.connect(self._on_import_all_magic_items_clicked)
            controls_layout.addWidget(filter_button)
            controls_layout.addWidget(import_button)
            controls_layout.addStretch()
            controls_group.setLayout(controls_layout)
            layout.addWidget(controls_group)
        elif name == "Armor":
            self.armor_table = table
            table.itemDoubleClicked.connect(self._show_armor_details)
            controls_group = QGroupBox("Armor Database")
            controls_layout = QHBoxLayout()
            filter_button = QPushButton("Advanced Filter...")
            filter_button.clicked.connect(self._open_armor_filter_dialog)
            import_button = QPushButton("Import All SRD Armor")
            import_button.setObjectName("import_all_button")
            import_button.clicked.connect(self._on_import_all_armor_clicked)
            controls_layout.addWidget(filter_button)
            controls_layout.addWidget(import_button)
            controls_layout.addStretch()
            controls_group.setLayout(controls_layout)
            layout.addWidget(controls_group)
        elif name == "Weapons":
            self.weapon_table = table
            table.itemDoubleClicked.connect(self._show_weapon_details)
            controls_group = QGroupBox("Weapon Database")
            controls_layout = QHBoxLayout()
            filter_button = QPushButton("Advanced Filter...")
            filter_button.clicked.connect(self._open_weapon_filter_dialog)
            import_button = QPushButton("Import All SRD Weapons")
            import_button.setObjectName("import_all_button")
            import_button.clicked.connect(self._on_import_all_weapons_clicked)
            controls_layout.addWidget(filter_button)
            controls_layout.addWidget(import_button)
            controls_layout.addStretch()
            controls_group.setLayout(controls_layout)
            layout.addWidget(controls_group)
        elif name == "Spells":
            self.spell_table = table
            table.itemDoubleClicked.connect(self._show_spell_details)
            controls_group = QGroupBox("Spell Database")
            controls_layout = QHBoxLayout()
            self.spell_search_input = QLineEdit()
            self.spell_search_input.setPlaceholderText("Search local spells by name...")
            search_button = QPushButton("Search")
            search_button.clicked.connect(self._on_local_spell_search_clicked)
            filter_button = QPushButton("Advanced Filter...")
            filter_button.clicked.connect(self._open_spell_filter_dialog)
            import_button = QPushButton("Import All SRD Spells")
            import_button.setObjectName("import_all_button")
            import_button.clicked.connect(self._on_import_all_spells_clicked)
            controls_layout.addWidget(self.spell_search_input)
            controls_layout.addWidget(search_button)
            controls_layout.addWidget(filter_button)
            controls_layout.addWidget(import_button)
            controls_layout.addStretch()
            controls_group.setLayout(controls_layout)
            layout.addWidget(controls_group)

        layout.addWidget(table)  # Add the table to the layout AFTER all controls

        if data:
            table.setRowCount(len(data))
            for row_num, row_data in enumerate(data):
                for col_num, header in enumerate(headers):
                    table.setItem(row_num, col_num, QTableWidgetItem(str(row_data.get(header.lower(), ""))))

        return widget

    def _populate_spell_table(self, spells):
        self.spell_table.setRowCount(0)
        self.spell_table.setRowCount(len(spells))
        for row_num, spell in enumerate(spells):
            self.spell_table.setItem(row_num, 0, QTableWidgetItem(spell.name))
            self.spell_table.setItem(row_num, 1, QTableWidgetItem(spell.level_str))
            self.spell_table.setItem(row_num, 2, QTableWidgetItem(spell.school))

    def _populate_armor_table(self, armors):
        self.armor_table.setRowCount(0)
        self.armor_table.setRowCount(len(armors))
        for row_num, armor in enumerate(armors):
            self.armor_table.setItem(row_num, 0, QTableWidgetItem(armor.name))
            self.armor_table.setItem(row_num, 1, QTableWidgetItem(armor.category))
            self.armor_table.setItem(row_num, 2, QTableWidgetItem(armor.ac_string))

    def _populate_weapon_table(self, weapons):
        self.weapon_table.setRowCount(0)
        self.weapon_table.setRowCount(len(weapons))
        for row_num, weapon in enumerate(weapons):
            self.weapon_table.setItem(row_num, 0, QTableWidgetItem(weapon.name))
            self.weapon_table.setItem(row_num, 1, QTableWidgetItem(weapon.category))
            self.weapon_table.setItem(row_num, 2, QTableWidgetItem(f"{weapon.damage_dice} {weapon.damage_type}"))

    def _populate_monster_table(self, monsters):
        self.monster_table.setRowCount(0)
        self.monster_table.setRowCount(len(monsters))
        for row_num, monster in enumerate(monsters):
            self.monster_table.setItem(row_num, 0, QTableWidgetItem(monster.name))
            self.monster_table.setItem(row_num, 1, QTableWidgetItem(monster.cr))
            self.monster_table.setItem(row_num, 2, QTableWidgetItem(monster.hp))
            self.monster_table.setItem(row_num, 3, QTableWidgetItem(monster.ac))

    def _populate_item_table(self, items):
        self.item_table.setRowCount(0)
        self.item_table.setRowCount(len(items))
        for row_num, item in enumerate(items):
            self.item_table.setItem(row_num, 0, QTableWidgetItem(item.name))
            self.item_table.setItem(row_num, 1, QTableWidgetItem(item.type))
            self.item_table.setItem(row_num, 2, QTableWidgetItem(item.rarity))

    def _open_item_filter_dialog(self):
        dialog = FilterMagicItemDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filters = dialog.get_filters()
            self._apply_item_filter(filters)

    def _open_weapon_filter_dialog(self):
        dialog = FilterWeaponDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filters = dialog.get_filters()
            self._apply_weapon_filter(filters)

    def _open_armor_filter_dialog(self):
        dialog = FilterArmorDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filters = dialog.get_filters()
            self._apply_armor_filter(filters)

    def _open_spell_filter_dialog(self):
        dialog = FilterSpellDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filters = dialog.get_filters()
            self._apply_spell_filter(filters)

    def _apply_spell_filter(self, filters):
        db_session = SessionLocal()
        try:
            query = db_session.query(Spell)
            if not filters:
                results = query.order_by(Spell.level_int, Spell.name).all()
            else:
                for key, value in filters.items():
                    query = query.filter(getattr(Spell, key) == value)
                results = query.order_by(Spell.level_int, Spell.name).all()
            self._populate_spell_table(results)
            self.statusBar().showMessage(f"Found {len(results)} spells matching filter.")
        finally:
            db_session.close()

    def _apply_armor_filter(self, filters):
        db_session = SessionLocal()
        try:
            query = db_session.query(Armor)
            if not filters:
                results = query.order_by(Armor.name).all()
            else:
                for key, value in filters.items():
                    query = query.filter(getattr(Armor, key) == value)
                results = query.order_by(Armor.name).all()
            self._populate_armor_table(results)
            self.statusBar().showMessage(f"Found {len(results)} armors matching filter.")
        finally:
            db_session.close()

    def _apply_weapon_filter(self, filters):
        db_session = SessionLocal()
        try:
            query = db_session.query(Weapon)
            if not filters:
                results = query.order_by(Weapon.name).all()
            else:
                for key, value in filters.items():
                    query = query.filter(getattr(Weapon, key) == value)
                results = query.order_by(Weapon.name).all()
            self._populate_weapon_table(results)
            self.statusBar().showMessage(f"Found {len(results)} weapons matching filter.")
        finally:
            db_session.close()

    def _on_import_all_armor_clicked(self):
        reply = QMessageBox.question(self, "Bulk Import",
                                     "This will fetch all armor from the Open5e SRD API. Continue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        self.armor_tab_content.findChild(QPushButton, "import_all_button").setEnabled(False)
        self.armor_importer_worker = ArmorImporterWorker()
        self.armor_importer_worker.progress.connect(self.statusBar().showMessage)
        self.armor_importer_worker.finished.connect(self._on_armor_import_finished)
        self.armor_importer_worker.start()

    def _on_armor_import_finished(self, count, error_msg):
        if error_msg:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{error_msg}")
            self.statusBar().showMessage("Import failed.")
        else:
            QMessageBox.information(self, "Import Complete", f"Successfully imported {count} new armors.")
            self.statusBar().showMessage("Import complete.")
            self._refresh_armor_tab()
        self.armor_tab_content.findChild(QPushButton, "import_all_button").setEnabled(True)

    def _on_import_all_spells_clicked(self):
        reply = QMessageBox.question(self, "Bulk Import",
                                     "This will fetch all spells from the Open5e SRD API. Continue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.spell_tab_content.findChild(QPushButton, "import_all_button").setEnabled(False)
        self.spell_importer_worker = SpellImporterWorker()
        self.spell_importer_worker.progress.connect(self.statusBar().showMessage)
        self.spell_importer_worker.finished.connect(self._on_spell_import_finished)
        self.spell_importer_worker.start()

    def _on_spell_import_finished(self, count, error_msg):
        if error_msg:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{error_msg}")
            self.statusBar().showMessage("Import failed.")
        else:
            QMessageBox.information(self, "Import Complete", f"Successfully imported {count} new spells.")
            self.statusBar().showMessage("Import complete.")
            self._refresh_spell_tab()

        self.spell_tab_content.findChild(QPushButton, "import_all_button").setEnabled(True)

    def _show_armor_details(self, item: QTableWidgetItem):
        row = item.row()
        item_name_item = self.armor_table.item(row, 0)
        if not item_name_item:
            return
        item_name = item_name_item.text()
        db_session = SessionLocal()
        try:
            armor_item = db_session.query(Armor).filter(Armor.name == item_name).first()
            if armor_item:
                dialog = ArmorDetailDialog(armor_item, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find details for '{item_name}' in the database.")
        finally:
            db_session.close()

    def _show_spell_details(self, item: QTableWidgetItem):
        item_table = self.sender()
        row = item.row()
        item_name_item = item_table.item(row, 0)
        if not item_name_item:
            return
        item_name = item_name_item.text()
        db_session = SessionLocal()
        try:
            spell_item = db_session.query(Spell).filter(Spell.name == item_name).first()
            if spell_item:
                dialog = SpellDetailDialog(spell_item, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find details for '{item_name}' in the database.")
        finally:
            db_session.close()

    def _on_import_all_weapons_clicked(self):
        reply = QMessageBox.question(self, "Bulk Import",
                                     "This will fetch all weapons from the Open5e SRD API. Continue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.weapon_tab_content.findChild(QPushButton, "import_all_button").setEnabled(False)
        self.weapon_importer_worker = WeaponImporterWorker()
        self.weapon_importer_worker.progress.connect(self.statusBar().showMessage)
        self.weapon_importer_worker.finished.connect(self._on_weapon_import_finished)
        self.weapon_importer_worker.start()

    def _on_weapon_import_finished(self, count, error_msg):
        if error_msg:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{error_msg}")
            self.statusBar().showMessage("Import failed.")
        else:
            QMessageBox.information(self, "Import Complete", f"Successfully imported {count} new weapons.")
            self.statusBar().showMessage("Import complete.")
            self._refresh_weapon_tab()

        self.weapon_tab_content.findChild(QPushButton, "import_all_button").setEnabled(True)

    def _apply_item_filter(self, filters):
        db_session = SessionLocal()
        try:
            query = db_session.query(MagicItem)
            if not filters:
                results = query.order_by(MagicItem.name).all()
            else:
                for key, value in filters.items():
                    if key == 'requires_attunement':
                        if value:
                            query = query.filter(
                                MagicItem.requires_attunement.isnot(None) & (MagicItem.requires_attunement != ""))
                        else:
                            query = query.filter(
                                MagicItem.requires_attunement.is_(None) | (MagicItem.requires_attunement == ""))
                    else:
                        query = query.filter(getattr(MagicItem, key) == value)
                results = query.order_by(MagicItem.name).all()

            self._populate_item_table(results)
            self.statusBar().showMessage(f"Found {len(results)} items matching filter.")
        finally:
            db_session.close()

    def _open_filter_dialog(self):
        dialog = FilterMonsterDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filters = dialog.get_filters()
            self._apply_monster_filter(filters)

    def _apply_monster_filter(self, filters):
        db_session = SessionLocal()
        try:
            query = db_session.query(Monster)
            if not filters:
                results = query.order_by(Monster.name).all()
            else:
                for key, value in filters.items():
                    query = query.filter(getattr(Monster, key) == value)
                results = query.order_by(Monster.name).all()

            self._populate_monster_table(results)
            self.statusBar().showMessage(f"Found {len(results)} monsters matching filter.")
        finally:
            db_session.close()

    def _show_weapon_details(self, item: QTableWidgetItem):
        item_table = self.sender()
        row = item.row()
        item_name_item = item_table.item(row, 0)
        if not item_name_item:
            return
        item_name = item_name_item.text()
        db_session = SessionLocal()
        try:
            weapon_item = db_session.query(Weapon).filter(Weapon.name == item_name).first()
            if weapon_item:
                dialog = WeaponDetailDialog(weapon_item, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find details for '{item_name}' in the database.")
        finally:
            db_session.close()

    def _open_new_monster_dialog(self):
        dialog = AddMonsterDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "Input Error", "Monster name cannot be empty.")
                return
            new_monster = Monster(**data)
            db_session = SessionLocal()
            try:
                db_session.add(new_monster)
                db_session.commit()
                self.statusBar().showMessage(f"Monster '{data['name']}' saved successfully.")
            except Exception as e:
                db_session.rollback()
                QMessageBox.critical(self, "Database Error", f"Could not save monster:\n{e}")
            finally:
                db_session.close()
            self._refresh_monster_tab()

    def _on_local_search_clicked(self):
        search_term = self.monster_search_input.text().strip()
        db_session = SessionLocal()
        try:
            if not search_term:
                results = db_session.query(Monster).order_by(Monster.name).all()
            else:
                search_pattern = f"%{search_term}%"
                results = db_session.query(Monster).filter(Monster.name.ilike(search_pattern)).order_by(
                    Monster.name).all()
            self._populate_monster_table(results)
            self.statusBar().showMessage(f"Found {len(results)} matching monsters.")
        finally:
            db_session.close()

    def _on_local_spell_search_clicked(self):
        """Handles searching the local database for spells."""
        search_term = self.spell_search_input.text().strip()
        db_session = SessionLocal()
        try:
            if not search_term:
                results = db_session.query(Spell).order_by(Spell.level_int, Spell.name).all()
            else:
                search_pattern = f"%{search_term}%"
                results = db_session.query(Spell).filter(Spell.name.ilike(search_pattern)).order_by(Spell.level_int,
                                                                                                    Spell.name).all()
            self._populate_spell_table(results)
            self.statusBar().showMessage(f"Found {len(results)} matching spells.")
        finally:
            db_session.close()

    def _on_import_all_monsters_clicked(self):
        reply = QMessageBox.question(self, "Bulk Import",
                                     "This will fetch all SRD monsters from the Open5e API. "
                                     "This may take a minute. Continue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.monster_tab_content.findChild(QPushButton, "import_all_button").setEnabled(False)
        self.monster_importer_worker = MonsterImporterWorker()
        self.monster_importer_worker.progress.connect(self.statusBar().showMessage)
        self.monster_importer_worker.finished.connect(self._on_import_finished)
        self.monster_importer_worker.start()

    def _on_import_all_magic_items_clicked(self):
        reply = QMessageBox.question(self, "Bulk Import",
                                     "This will fetch all magic items from the Open5e SRD API. Continue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.item_tab_content.findChild(QPushButton, "import_all_button").setEnabled(False)
        self.magic_item_importer_worker = MagicItemImporterWorker()
        self.magic_item_importer_worker.progress.connect(self.statusBar().showMessage)
        self.magic_item_importer_worker.finished.connect(self._on_magic_item_import_finished)
        self.magic_item_importer_worker.start()

    def _on_magic_item_import_finished(self, count, error_msg):
        if error_msg:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{error_msg}")
            self.statusBar().showMessage("Import failed.")
        else:
            QMessageBox.information(self, "Import Complete", f"Successfully imported {count} new magic items.")
            self.statusBar().showMessage("Import complete.")
            self._refresh_item_tab()

        self.item_tab_content.findChild(QPushButton, "import_all_button").setEnabled(True)

    def _on_import_finished(self, count, error_msg):
        if error_msg:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import:\n{error_msg}")
            self.statusBar().showMessage("Import failed.")
        else:
            QMessageBox.information(self, "Import Complete", f"Successfully imported {count} new monsters.")
            self.statusBar().showMessage("Import complete.")
            self._refresh_monster_tab()

        self.monster_tab_content.findChild(QPushButton, "import_all_button").setEnabled(True)

    def _show_monster_details(self, item: QTableWidgetItem):
        monster_table = self.sender()
        if not isinstance(monster_table, QTableWidget):
            return

        row = item.row()
        monster_name_item = monster_table.item(row, 0)
        if not monster_name_item:
            return

        monster_name = monster_name_item.text()

        db_session = SessionLocal()
        try:
            monster = db_session.query(Monster).filter(Monster.name == monster_name).first()
            if monster:
                dialog = MonsterDetailDialog(monster, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find details for '{monster_name}' in the database.")
        finally:
            db_session.close()

    def _show_magic_item_details(self, item: QTableWidgetItem):
        item_table = self.sender()
        if not isinstance(item_table, QTableWidget):
            return
        row = item.row()
        item_name_item = item_table.item(row, 0)
        if not item_name_item:
            return
        item_name = item_name_item.text()
        db_session = SessionLocal()
        try:
            magic_item = db_session.query(MagicItem).filter(MagicItem.name == item_name).first()
            if magic_item:
                dialog = MagicItemDetailDialog(magic_item, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Not Found", f"Could not find details for '{item_name}' in the database.")
        finally:
            db_session.close()

    # --- Combat Tracker Methods ---
    def add_combatant(self, name, hp, initiative=None):
        """Add a combatant to the initiative table.

        Args:
            name (str): Combatant's name.
            hp (int): Combatant's hit points.
            initiative (int, optional): Combatant's initiative roll.
        """
        logging.debug(f"Adding combatant: {name}, HP: {hp}, Initiative: {initiative}")

        try:
            if not hasattr(self, "initiative_table") or self.initiative_table is None:
                logging.error("Initiative table not initialized")
                return

            if initiative is None:
                initiative = random.randint(1, 20)

            row_position = self.initiative_table.rowCount()
            self.initiative_table.insertRow(row_position)

            name_item = QTableWidgetItem(name)
            init_item = QTableWidgetItem()
            init_item.setData(Qt.ItemDataRole.DisplayRole, initiative)
            hp_item = QTableWidgetItem(str(hp))

            self.initiative_table.setItem(row_position, 0, name_item)
            self.initiative_table.setItem(row_position, 1, init_item)
            self.initiative_table.setItem(row_position, 2, hp_item)

            self.ailments[name] = []  # Initialize ailments for combatant
            self._append_to_combat_log(f"Added {name} (HP: {hp}, Initiative: {initiative}) to combat.")

            logging.debug(f"Combatant {name} added successfully")
        except Exception as e:
            logging.error(f"Error adding combatant {name}: {str(e)}")
            self._append_to_combat_log(f"Error adding {name}: {str(e)}")

    def _add_combatant_from_input(self):
        """Add a combatant from user input fields."""
        name = self.name_input.text().strip()
        hp = self.hp_input.text().strip()
        initiative = self.initiative_input.text().strip()

        if not name or not hp:
            QMessageBox.warning(self, "Input Error", "Name and HP are required.")
            return

        try:
            hp = int(hp)
            initiative = int(initiative) if initiative else None
            self.add_combatant(name, hp, initiative)
            self.name_input.clear()
            self.hp_input.clear()
            self.initiative_input.clear()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "HP and Initiative must be numeric.")

    def add_combatant(self, name, hp, initiative=None):
        """Add a combatant to the initiative table.

        Args:
            name (str): Combatant's name.
            hp (int): Combatant's hit points.
            initiative (int, optional): Combatant's initiative roll.
        """
        logging.debug(f"Adding combatant: {name}, HP: {hp}, Initiative: {initiative}")

        try:
            if not hasattr(self, "initiative_table") or self.initiative_table is None:
                logging.error("Initiative table not initialized")
                return

            if initiative is None:
                initiative = random.randint(1, 20)

            # Add to initiative table
            row_position = self.initiative_table.rowCount()
            self.initiative_table.insertRow(row_position)

            name_item = QTableWidgetItem(name)
            init_item = QTableWidgetItem()
            init_item.setData(Qt.ItemDataRole.DisplayRole, initiative)
            hp_item = QTableWidgetItem(str(hp))
            ailment_item = QTableWidgetItem("")

            self.initiative_table.setItem(row_position, 0, name_item)
            self.initiative_table.setItem(row_position, 1, init_item)
            self.initiative_table.setItem(row_position, 2, hp_item)
            self.initiative_table.setItem(row_position, 3, ailment_item)

            # Initialize ailments
            self.ailments[name] = []

            logging.debug(f"Combatant {name} added successfully")
        except Exception as e:
            logging.error(f"Error adding combatant {name}: {str(e)}")

    def _remove_selected_combatant(self):
        """Remove selected combatants from the initiative table."""
        selected_rows = self.initiative_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "No combatant selected to remove.")
            return

        combatant_names_to_remove = []
        for index in selected_rows:
            name_item = self.initiative_table.item(index.row(), 0)
            if name_item:
                combatant_names_to_remove.append(name_item.text())

        for index in sorted([index.row() for index in selected_rows], reverse=True):
            self.initiative_table.removeRow(index)

        for name in combatant_names_to_remove:
            self._remove_ailments_by_target(name)
            self._append_to_combat_log(f"Removed {name} from combat.")

        if self.initiative_table.rowCount() == 0:
            self.current_turn_row = -1
            self.current_round = 1
            self.round_counter_label.setText(f"Round: {self.current_round}")
            self.first_combatant_name_for_round_check = ""
        elif self.current_turn_row >= self.initiative_table.rowCount():
            self.current_turn_row = -1

        self._highlight_current_turn()


    def _edit_ailments(self, row, column):
        """Edit ailments for a combatant via double-click on the Ailments column."""
        if column != 3:  # Only edit on Ailments column
            return

        name_item = self.initiative_table.item(row, 0)
        if not name_item:
            return

        name = name_item.text()
        current_ailments = self.ailments.get(name, [])
        ailments_str = ", ".join(current_ailments)

        # Prompt user to edit ailments
        new_ailments, ok = QInputDialog.getText(
            self,
            "Edit Ailments",
            f"Enter ailments for {name} (comma-separated):",
            text=ailments_str,
        )

        if ok:
            # Parse new ailments
            ailments = [a.strip() for a in new_ailments.split(",") if a.strip()]
            self.ailments[name] = ailments
            ailment_item = QTableWidgetItem(", ".join(ailments))
            self.initiative_table.setItem(row, 3, ailment_item)
            logging.debug(f"Updated ailments for {name}: {ailments}")

    def _remove_ailments_by_target(self, name):
        """Remove all ailments associated with a combatant.

        Args:
            name (str): Combatant's name.
        """
        if name in self.ailments:
            del self.ailments[name]
            logging.debug(f"Removed ailments for {name}")

    def _update_combatant_hp(self, action_type):
        selected_rows = self.initiative_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "No combatant selected to update HP.")
            return

        row = selected_rows[0].row()
        name_item = self.initiative_table.item(row, 0)
        hp_item = self.initiative_table.item(row, 2)

        if not name_item or not hp_item:
            QMessageBox.critical(self, "Error", "Could not retrieve combatant data.")
            return

        try:
            current_hp = int(hp_item.text())
        except ValueError:
            QMessageBox.critical(self, "Data Error",
                                 f"HP value '{hp_item.text()}' for {name_item.text()} is not a valid number. Please correct it.")
            return

        combatant_name = name_item.text()

        min_val = 1
        max_val = 99999

        amount, ok = QInputDialog.getInt(self,
                                         f"{'Deal Damage' if action_type == 'damage' else 'Heal'}",
                                         f"Enter amount for {combatant_name}:",
                                         1,
                                         min_val,
                                         max_val)

        if ok:
            new_hp = current_hp - amount if action_type == "damage" else current_hp + amount
            if action_type == "damage":
                new_hp = max(0, new_hp)

            hp_item.setText(str(new_hp))
            self._append_to_combat_log(
                f"{combatant_name} {'took' if action_type == 'damage' else 'healed'} {amount} HP. Current HP: {new_hp}.")
            if new_hp <= 0 and action_type == "damage":
                QMessageBox.information(self, "Combatant Down", f"{combatant_name} has fallen!")
                self._append_to_combat_log(f"{combatant_name} is at 0 HP or less!")

    def _sort_initiative(self):
        self.initiative_table.sortItems(1, Qt.SortOrder.DescendingOrder)
        self._append_to_combat_log("Initiative order sorted.")

        if self.initiative_table.rowCount() > 0:
            first_combatant_item = self.initiative_table.item(0, 0)
            if first_combatant_item:
                self.first_combatant_name_for_round_check = first_combatant_item.text()
        else:
            self.first_combatant_name_for_round_check = ""

        self.current_turn_row = -1
        self._highlight_current_turn()

    def _next_turn(self):
        """Advance to the next turn in combat.

        TODO: Implement ailment duration tracking if needed.
        """
        if self.initiative_table.rowCount() == 0:
            return

        self.current_turn_row = (
            self.current_turn_row + 1
        ) % self.initiative_table.rowCount()
        if self.current_turn_row == 0:
            self.current_round += 1
            self.round_counter_label.setText(f"Round: {self.current_round}")
        self._highlight_current_turn()
        logging.debug(
            f"Advanced to turn {self.current_turn_row}, round {self.current_round}"
        )

    def _highlight_current_turn(self):
        """Highlight the current combatant's turn in the initiative table."""
        for row in range(self.initiative_table.rowCount()):
            for col in range(self.initiative_table.columnCount()):
                item = self.initiative_table.item(row, col)
                if item:
                    color = (
                        Qt.GlobalColor.white
                        if row != self.current_turn_row
                        else Qt.GlobalColor.lightGray
                    )
                    item.setBackground(color)

    def _generate_encounter(self):
        """Generate an encounter and add monsters to the combat tracker."""
        self._place_entities()  # Use map-based entity placement
        logging.debug("Generated encounter via _place_entities")

    def _append_to_combat_log(self, message):
        """Append a message to the combat log."""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.combat_log.append(f"{timestamp} {message}")
        self.combat_log.verticalScrollBar().setValue(self.combat_log.verticalScrollBar().maximum())
        logging.debug(message)

    def _clear_highlight(self):
        pass

    def _append_to_combat_log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.combat_log.append(f"{timestamp} {message}")
        self.combat_log.verticalScrollBar().setValue(self.combat_log.verticalScrollBar().maximum())

    def _add_ailment(self):
        """Add an ailment to the ailment table."""
        target = self.ailment_target_input.text().strip()
        ailment_name = self.ailment_name_input.text().strip()
        duration_text = self.ailment_duration_input.text().strip()
        source = "DM"

        if not target:
            QMessageBox.warning(self, "Input Error", "Ailment target name cannot be empty.")
            return
        if not ailment_name:
            QMessageBox.warning(self, "Input Error", "Ailment name cannot be empty.")
            return
        if not duration_text.isdigit() or int(duration_text) <= 0:
            QMessageBox.warning(self, "Input Error", "Duration must be a positive number of turns.")
            return

        duration = int(duration_text)

        row_position = self.ailment_table.rowCount()
        self.ailment_table.insertRow(row_position)
        self.ailment_table.setItem(row_position, 0, QTableWidgetItem(target))
        self.ailment_table.setItem(row_position, 1, QTableWidgetItem(ailment_name))
        duration_item = QTableWidgetItem()
        duration_item.setData(Qt.ItemDataRole.DisplayRole, duration)
        self.ailment_table.setItem(row_position, 2, duration_item)
        self.ailment_table.setItem(row_position, 3, QTableWidgetItem(source))

        self.ailments[target].append({"name": ailment_name, "duration": duration, "source": source})
        self._append_to_combat_log(
            f"Added ailment '{ailment_name}' to {target} for {duration} turns (Source: {source}).")

        self.ailment_target_input.clear()
        self.ailment_name_input.clear()
        self.ailment_duration_input.clear()
    def _remove_selected_ailment(self):
        """Remove selected ailments from the ailment table."""
        selected_rows = self.ailment_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "No ailment selected to remove.")
            return

        for index in sorted([index.row() for index in selected_rows], reverse=True):
            target_item = self.ailment_table.item(index.row(), 0)
            ailment_item = self.ailment_table.item(index.row(), 1)
            if target_item and ailment_item:
                target_name = target_item.text()
                ailment_name = ailment_item.text()
                self.ailments[target_name] = [a for a in self.ailments[target_name] if a["name"] != ailment_name]
                self._append_to_combat_log(f"Manually removed ailment '{ailment_name}' from {target_name}.")
            self.ailment_table.removeRow(index)


    def _remove_ailments_by_target(self, target_name):
        """Remove all ailments associated with a specific target combatant."""
        rows_to_remove = []
        for r in range(self.ailment_table.rowCount()):
            target_item = self.ailment_table.item(r, 0)
            if target_item and target_item.text() == target_name:
                rows_to_remove.append(r)

        for row_index in sorted(rows_to_remove, reverse=True):
            ailment_item = self.ailment_table.item(row_index, 1)
            if ailment_item:
                self._append_to_combat_log(
                    f"Removed ailment '{ailment_item.text()}' from {target_name} due to combatant removal.")
            self.ailment_table.removeRow(row_index)

        if target_name in self.ailments:
            del self.ailments[target_name]

    def _decrement_ailment_duration(self, current_combatant_name):
        """Decrement ailment durations for a combatant and remove expired ailments."""
        rows_to_remove = []
        for r in range(self.ailment_table.rowCount()):
            target_item = self.ailment_table.item(r, 0)
            duration_item = self.ailment_table.item(r, 2)
            ailment_name_item = self.ailment_table.item(r, 1)

            if target_item and duration_item and ailment_name_item and target_item.text() == current_combatant_name:
                try:
                    current_duration = duration_item.data(Qt.ItemDataRole.DisplayRole)
                    if current_duration is None:
                        current_duration = int(duration_item.text())

                    if current_duration > 0:
                        new_duration = current_duration - 1
                        duration_item.setData(Qt.ItemDataRole.DisplayRole, new_duration)
                        duration_item.setText(str(new_duration))
                        self.ailments[current_combatant_name] = [
                            a for a in self.ailments[current_combatant_name]
                            if a["name"] != ailment_name_item.text() or a["duration"] != current_duration
                        ]
                        if new_duration > 0:
                            self.ailments[current_combatant_name].append({
                                "name": ailment_name_item.text(),
                                "duration": new_duration,
                                "source": self.ailment_table.item(r, 3).text()
                            })
                            self._append_to_combat_log(
                                f"'{ailment_name_item.text()}' on {current_combatant_name} now has {new_duration} turns remaining.")
                        if new_duration == 0:
                            rows_to_remove.append(r)
                            self._append_to_combat_log(
                                f"'{ailment_name_item.text()}' on {current_combatant_name} has ended.")
                    else:
                        rows_to_remove.append(r)
                except ValueError:
                    self._append_to_combat_log(
                        f"Error: Invalid duration for ailment '{ailment_name_item.text()}' on {current_combatant_name}. Marking for removal.")
                    rows_to_remove.append(r)

        for row_index in sorted(rows_to_remove, reverse=True):
            self.ailment_table.removeRow(row_index)

    def _next_turn(self):
        """Advance to the next turn in combat."""
        if self.initiative_table.rowCount() == 0:
            return

        self.current_turn_row = (self.current_turn_row + 1) % self.initiative_table.rowCount()
        if self.current_turn_row == 0:
            self.current_round += 1
            self.round_counter_label.setText(f"Round: {self.current_round}")
            self._append_to_combat_log(f"Round {self.current_round} begins.")

        name_item = self.initiative_table.item(self.current_turn_row, 0)
        if name_item:
            current_combatant = name_item.text()
            self._decrement_ailment_duration(current_combatant)
            self._append_to_combat_log(f"{current_combatant}'s turn.")
        self._highlight_current_turn()


    def _save_last_state_path(self, path):
        """Saves the path to the last loaded/saved combat state file."""
        try:
            with open(self.LAST_STATE_FILE, 'w', encoding='utf-8') as f:
                f.write(path)
        except IOError as e:
            print(f"Warning: Could not save last state path to '{self.LAST_STATE_FILE}': {e}")
            self.statusBar().showMessage(f"Warning: Could not save last state path.")

    def _load_last_state_path(self):
        """Loads the path of the last combat state file."""
        if os.path.exists(self.LAST_STATE_FILE):
            try:
                with open(self.LAST_STATE_FILE, 'r', encoding='utf-8') as f:
                    path = f.read().strip()
                    if path and os.path.exists(path):
                        return path
            except IOError as e:
                print(f"Error reading last state path from '{self.LAST_STATE_FILE}': {e}")
                self.statusBar().showMessage(f"Error loading last state path.")
        return None

    def _save_combat_state(self, file_path=None, is_auto_save=False):
        combatants_data = []
        for row in range(self.initiative_table.rowCount()):
            name = self.initiative_table.item(row, 0).text()
            initiative = self.initiative_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole)
            hp = self.initiative_table.item(row, 2).text()
            combatants_data.append({"name": name, "initiative": initiative, "hp": hp})

        ailments_data = []
        for row in range(self.ailment_table.rowCount()):
            target = self.ailment_table.item(row, 0).text()
            ailment = self.ailment_table.item(row, 1).text()
            duration = self.ailment_table.item(row, 2).data(Qt.ItemDataRole.DisplayRole)
            source = self.ailment_table.item(row, 3).text()
            ailments_data.append({"target": target, "ailment": ailment, "duration": duration, "source": source})

        combat_state = {
            "combatants": combatants_data,
            "ailments": ailments_data,
            "current_turn_row": self.current_turn_row,
            "current_round": self.current_round,
            "combat_log": self.combat_log.toPlainText(),
            "first_combatant_name_for_round_check": self.first_combatant_name_for_round_check
        }

        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Combat State", "",
                                                       "Combat State Files (*.json);;All Files (*)")
            if not file_path:
                self.statusBar().showMessage("Combat state save cancelled.")
                return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(combat_state, f, indent=4)

            if is_auto_save:
                self.statusBar().showMessage(f"Auto-saved combat state to '{os.path.basename(file_path)}'")
            else:
                self.statusBar().showMessage(f"Combat state saved to '{os.path.basename(file_path)}'")
                self._append_to_combat_log(f"Combat state saved to '{os.path.basename(file_path)}'")
                self.current_combat_file_path = file_path
                self._save_last_state_path(file_path)

        except Exception as e:
            msg = "Auto-save failed." if is_auto_save else "Failed to save combat state."
            QMessageBox.critical(self, "Save Error", f"{msg}:\n{e}")
            self.statusBar().showMessage(msg)

    def _auto_save_combat_state(self):
        """Automatically saves the combat state to the current combat file path."""
        if self.current_combat_file_path:
            self._save_combat_state(file_path=self.current_combat_file_path, is_auto_save=True)
        else:
            default_auto_save_path = "auto_save_combat_state.json"
            self._save_combat_state(file_path=default_auto_save_path, is_auto_save=True)
            self.current_combat_file_path = default_auto_save_path
            self._save_last_state_path(default_auto_save_path)

    def _load_combat_state(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Combat State", "",
                                                       "Combat State Files (*.json);;All Files (*)")
            if not file_path:
                self.statusBar().showMessage("Combat state load cancelled.")
                return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                combat_state = json.load(f)

            self._clear_highlight()
            self.initiative_table.setRowCount(0)
            self.ailment_table.setRowCount(0)
            self.combat_log.clear()

            combatants_data = combat_state.get("combatants", [])
            self.initiative_table.setRowCount(len(combatants_data))
            for row_num, combatant in enumerate(combatants_data):
                name_item = QTableWidgetItem(combatant.get("name", ""))
                init_item = QTableWidgetItem()
                init_item.setData(Qt.ItemDataRole.DisplayRole, int(combatant.get("initiative", 0)))
                hp_item = QTableWidgetItem(str(combatant.get("hp", "")))
                self.initiative_table.setItem(row_num, 0, name_item)
                self.initiative_table.setItem(row_num, 1, init_item)
                self.initiative_table.setItem(row_num, 2, hp_item)

            ailments_data = combat_state.get("ailments", [])
            self.ailment_table.setRowCount(len(ailments_data))
            for row_num, ailment in enumerate(ailments_data):
                target_item = QTableWidgetItem(ailment.get("target", ""))
                ailment_name_item = QTableWidgetItem(ailment.get("ailment", ""))
                duration_item = QTableWidgetItem()
                duration_item.setData(Qt.ItemDataRole.DisplayRole, int(ailment.get("duration", 0)))
                source_item = QTableWidgetItem(ailment.get("source", ""))
                self.ailment_table.setItem(row_num, 0, target_item)
                self.ailment_table.setItem(row_num, 1, ailment_name_item)
                self.ailment_table.setItem(row_num, 2, duration_item)
                self.ailment_table.setItem(row_num, 3, source_item)

            self.current_turn_row = combat_state.get("current_turn_row", -1)
            self.current_round = combat_state.get("current_round", 1)
            self.combat_log.setPlainText(combat_state.get("combat_log", "Combat log loaded."))
            self.first_combatant_name_for_round_check = combat_state.get("first_combatant_name_for_round_check", "")

            self.round_counter_label.setText(f"Round: {self.current_round}")
            self._highlight_current_turn()
            self.combat_log.verticalScrollBar().setValue(self.combat_log.verticalScrollBar().maximum())

            self.statusBar().showMessage(f"Combat state loaded from '{os.path.basename(file_path)}'")
            self._append_to_combat_log(f"Combat state loaded from '{os.path.basename(file_path)}'")
            self.current_combat_file_path = file_path
            self._save_last_state_path(file_path)

        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                                 f"Failed to load combat state from '{os.path.basename(file_path)}':\n{e}")
            self.statusBar().showMessage("Failed to load combat state.")

    def create_map_generator_tab(self):
        logging.debug("Creating map generator tab")
        map_tab = QWidget(self)
        layout = QHBoxLayout(map_tab)

        # Controls
        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.addWidget(QLabel("<h2>Map Generation</h2>", self))

        # Map Settings
        settings_group = QGroupBox("Map Settings")
        settings_layout = QVBoxLayout()
        size_layout = QHBoxLayout()
        self.map_width_input = QLineEdit(self)
        self.map_width_input.setPlaceholderText("Width (5-30)")
        self.map_width_input.setText("20")
        self.map_height_input = QLineEdit(self)
        self.map_height_input.setPlaceholderText("Height (5-30)")
        self.map_height_input.setText("10")
        size_layout.addWidget(QLabel("Grid Size:", self))
        size_layout.addWidget(self.map_width_input)
        size_layout.addWidget(self.map_height_input)
        settings_layout.addLayout(size_layout)

        theme_layout = QHBoxLayout()
        self.map_theme_combo = QComboBox(self)
        self.map_theme_combo.addItems(["Dungeon", "Cave", "Castle", "Forest"])
        theme_layout.addWidget(QLabel("Theme:", self))
        theme_layout.addWidget(self.map_theme_combo)
        settings_layout.addLayout(theme_layout)

        settings_group.setLayout(settings_layout)
        controls_layout.addWidget(settings_group)

        # Generation Controls
        generate_button = QPushButton("Generate Map", self)
        generate_button.clicked.connect(self._on_generate_map_clicked)
        controls_layout.addWidget(generate_button)

        save_button = QPushButton("Save Map as Image", self)
        save_button.clicked.connect(self._save_map_as_image)
        controls_layout.addWidget(save_button)

        # Entity Placement
        entity_group = QGroupBox("Place Entities")
        entity_layout = QVBoxLayout()

        # Entity Checkboxes and Quantities
        entity_types = ["Monsters", "Magic Items", "Armor", "Weapons"]
        self.entity_checkboxes = {}
        self.entity_quantity_inputs = {}
        for entity_type in entity_types:
            checkbox_layout = QHBoxLayout()
            checkbox = QCheckBox(entity_type, self)
            quantity_input = QLineEdit(self)
            quantity_input.setPlaceholderText(f"{entity_type} (0-5)")
            quantity_input.setText("0")
            quantity_input.setEnabled(False)
            checkbox.stateChanged.connect(
                lambda state, inp=quantity_input: inp.setEnabled(state == Qt.CheckState.Checked.value))
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.addWidget(quantity_input)
            entity_layout.addLayout(checkbox_layout)
            self.entity_checkboxes[entity_type] = checkbox
            self.entity_quantity_inputs[entity_type] = quantity_input

        # Party Level
        level_layout = QHBoxLayout()
        self.party_level_input = QLineEdit(self)
        self.party_level_input.setPlaceholderText("Party Level (1-20)")
        self.party_level_input.setText("5")
        level_layout.addWidget(QLabel("Party Level:", self))
        level_layout.addWidget(self.party_level_input)
        entity_layout.addLayout(level_layout)

        # Place Button
        place_button = QPushButton("Place Entities", self)
        place_button.clicked.connect(self._place_entities)
        entity_layout.addWidget(place_button)

        entity_group.setLayout(entity_layout)
        controls_layout.addWidget(entity_group)

        # Map Description
        desc_group = QGroupBox("Map Description")
        desc_layout = QVBoxLayout()
        self.map_description = QTextEdit(self)
        self.map_description.setReadOnly(True)
        self.map_description.setPlaceholderText("Map details will appear here...")
        desc_layout.addWidget(self.map_description)
        desc_group.setLayout(desc_layout)
        controls_layout.addWidget(desc_group)

        controls_layout.addStretch()

        # Map Display
        self.map_label = QLabel(self)
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setMinimumSize(300, 200)
        self.map_pixmap = None
        self.map_with_entities_pixmap = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls_widget)
        splitter.addWidget(self.map_label)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)
        self.central_widget.addTab(map_tab, "Map Generator")

        # Initialize map
        self._on_generate_map_clicked()
        logging.debug("Map generator tab created")

    def _place_entities(self):
        logging.debug("Starting _place_entities")
        if not self.current_map_rooms:
            QMessageBox.warning(self, "Placement Error", "No rooms available. Generate a map first.")
            self._append_to_combat_log("Error: No rooms available for entity placement.")
            logging.debug("No rooms available")
            return

        # Validate party level
        try:
            party_level = int(self.party_level_input.text())
            if not 1 <= party_level <= 20:
                raise ValueError("Party level must be between 1 and 20.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid party level: {str(e)}")
            self._append_to_combat_log(f"Error: Invalid party level: {str(e)}")
            logging.debug(f"Invalid party level: {str(e)}")
            return

        # Collect selected entity types and quantities
        entity_requests = {}
        total_quantity = 0
        for entity_type in self.entity_checkboxes:
            if self.entity_checkboxes[entity_type].isChecked():
                try:
                    qty = int(self.entity_quantity_inputs[entity_type].text())
                    if not 0 <= qty <= 5:  # Lower max per type
                        raise ValueError(f"Quantity for {entity_type} must be between 0 and 5.")
                    if qty > 0:
                        entity_requests[entity_type] = qty
                        total_quantity += qty
                except ValueError as e:
                    QMessageBox.warning(self, "Input Error", str(e))
                    self._append_to_combat_log(f"Error: {str(e)}")
                    logging.debug(f"Invalid quantity for {entity_type}: {str(e)}")
                    return

        if not entity_requests:
            QMessageBox.warning(self, "Selection Error",
                                "Please select at least one entity type with a quantity greater than 0.")
            self._append_to_combat_log("Error: No entity types selected for placement.")
            logging.debug("No entity types selected")
            return

        logging.debug(f"Entity requests: {entity_requests}, total: {total_quantity}")

        # Cap total entities to available rooms
        max_entities = min(total_quantity, len(self.current_map_rooms), 20)  # Hard cap at 20
        if max_entities < total_quantity:
            self._append_to_combat_log(
                f"Warning: Limited to {max_entities} entities due to {len(self.current_map_rooms)} rooms available.")
            logging.debug(f"Limited to {max_entities} entities")
            total_quantity = max_entities

        # Fetch entities
        available_entities = {}
        db_session = SessionLocal()
        try:
            for entity_type, qty in entity_requests.items():
                logging.debug(f"Querying {entity_type}")
                if entity_type == "Monsters":
                    cr_max = max(1, party_level // 2)
                    query = db_session.query(Monster).filter(Monster.cr.in_([str(i) for i in range(1, cr_max + 1)]))
                    entities = query.limit(50).all()  # Lower limit
                elif entity_type == "Magic Items":
                    rarities = ["Common", "Uncommon", "Rare"] if party_level < 10 else ["Common", "Uncommon", "Rare",
                                                                                        "Very Rare"]
                    query = db_session.query(MagicItem).filter(MagicItem.rarity.in_(rarities))
                    entities = query.limit(50).all()
                elif entity_type == "Armor":
                    query = db_session.query(Armor)
                    entities = query.limit(50).all()
                elif entity_type == "Weapons":
                    query = db_session.query(Weapon)
                    entities = query.limit(50).all()

                if not entities:
                    QMessageBox.warning(self, "Placement Error", f"No {entity_type} found in database.")
                    self._append_to_combat_log(f"Error: No {entity_type} found in database.")
                    logging.debug(f"No {entity_type} found")
                    return
                available_entities[entity_type] = entities
                self._append_to_combat_log(f"Found {len(entities)} {entity_type} for placement.")
                logging.debug(f"Found {len(entities)} {entity_type}")
        except Exception as e:
            self._append_to_combat_log(f"Database error: {str(e)}")
            logging.error(f"Database error: {str(e)}")
            return
        finally:
            db_session.close()
            logging.debug("Database session closed")

        # Clear existing entities
        self.active_map_entities.clear()
        logging.debug("Cleared active_map_entities")

        # Initialize pixmap
        try:
            self.map_with_entities_pixmap = self.map_pixmap.copy() if self.map_pixmap else None
            if self.map_with_entities_pixmap is None or self.map_with_entities_pixmap.isNull():
                raise ValueError("Invalid map pixmap")
        except Exception as e:
            QMessageBox.critical(self, "Pixmap Error", f"Failed to copy map pixmap: {str(e)}")
            self._append_to_combat_log(f"Error copying pixmap: {str(e)}")
            logging.error(f"Pixmap error: {str(e)}")
            return

        # Initialize painter
        painter = None
        try:
            painter = QPainter(self.map_with_entities_pixmap)
            tile_size = 10  # Smaller tiles
            font = QFont("Arial", 6)  # Smaller font
            painter.setFont(font)
        except Exception as e:
            QMessageBox.critical(self, "Painter Error", f"Failed to initialize QPainter: {str(e)}")
            self._append_to_combat_log(f"Error initializing QPainter: {str(e)}")
            logging.error(f"Painter error: {str(e)}")
            return

        # Color coding
        colors = {
            "Monsters": QColor("#FF0000"),
            "Magic Items": QColor("#00FF00"),
            "Armor": QColor("#0000FF"),
            "Weapons": QColor("#FFFF00")
        }

        # Prepare placement list
        placement_list = []
        for entity_type, qty in entity_requests.items():
            placement_list.extend([(entity_type, qty) for _ in range(qty)])
        random.shuffle(placement_list)
        logging.debug(f"Placement list: {len(placement_list)} items")

        # Place entities
        placed = []
        used_rooms = set()
        try:
            for i, (entity_type, _) in enumerate(placement_list[:max_entities]):
                logging.debug(f"Placing {entity_type} #{i + 1}")
                entities = available_entities[entity_type]
                entity = random.choice(entities)
                available_rooms = [r for j, r in enumerate(self.current_map_rooms) if j not in used_rooms]
                if not available_rooms:
                    available_rooms = self.current_map_rooms
                room = random.choice(available_rooms)
                room_idx = self.current_map_rooms.index(room)
                used_rooms.add(room_idx)
                rx, ry, rw, rh = room

                # Validate placement
                if rw <= 2 or rh <= 2:
                    self._append_to_combat_log(f"Skipping invalid room {room_idx + 1} with size {rw}x{rh}")
                    logging.debug(f"Invalid room {room_idx + 1}: {rw}x{rh}")
                    continue

                px = random.randint(rx + 1, rx + rw - 2)
                py = random.randint(ry + 1, ry + rh - 2)
                logging.debug(f"Position for {entity_type}: ({px}, {py})")

                # Draw entity
                painter.setBrush(QBrush(colors[entity_type]))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(px * tile_size, py * tile_size, tile_size, tile_size)
                painter.setPen(QPen(QColor("#FFFFFF")))
                name = getattr(entity, "name", "X")[:1]
                painter.drawText(px * tile_size, py * tile_size, tile_size, tile_size,
                                 Qt.AlignmentFlag.AlignCenter, name)
                logging.debug(f"Drew {entity_type} '{name}' at ({px}, {py})")

                # Store entity
                self.active_map_entities.append({
                    "type": entity_type,
                    "data": entity,
                    "map_x": px,
                    "map_y": py,
                    "room": room
                })
                placed.append((entity_type, entity, px, py, room))

                # Add monsters to combat tracker
                if entity_type == "Monsters":
                    try:
                        name = getattr(entity, "name", "Unknown Monster")
                        hp = getattr(entity, "hp", 10)
                        if not isinstance(hp, (int, float)):
                            hp = 10
                        self.add_combatant(name=name, hp=hp)
                        self._append_to_combat_log(f"Added {name} (HP: {hp}) to combat tracker.")
                        logging.debug(f"Added {name} (HP: {hp}) to combat tracker")
                    except Exception as e:
                        self._append_to_combat_log(f"Error adding {name} to combat tracker: {str(e)}")
                        logging.error(f"Combat tracker error for {name}: {str(e)}")

                QApplication.processEvents()  # Keep UI responsive
        except Exception as e:
            self._append_to_combat_log(f"Placement error: {str(e)}")
            logging.error(f"Placement error: {str(e)}")
        finally:
            if painter:
                painter.end()
            logging.debug("Painter ended")

        # Update description
        try:
            desc = self.map_description.toPlainText().split("\nPlaced Entities:")[0]
            desc += "\nPlaced Entities:\n"
            for i, (entity_type, entity, _, x, y, room) in enumerate(placed, 1):
                name = getattr(entity, "name", "Unknown")
                desc += f"{i}. {entity_type}: {name} at ({x},{y}) in Room {self.current_map_rooms.index(room) + 1}\n"
            self.map_description.setPlainText(desc)
        except Exception as e:
            self._append_to_combat_log(f"Description update error: {str(e)}")
            logging.error(f"Description error: {str(e)}")

        self._update_map_display()
        self._append_to_combat_log(f"Placed {len(placed)} entities")
        self.statusBar().showMessage(f"Placed {len(placed)} entities on the map.")
        logging.debug(f"Completed placement: {len(placed)} entities")

    def _place_entities(self):
        if not self.current_map_rooms:
            QMessageBox.warning(self, "Placement Error", "No rooms available. Generate a map first.")
            self._append_to_combat_log("Error: No rooms available for entity placement.")
            return

        # Validate party level
        try:
            party_level = int(self.party_level_input.text())
            if not 1 <= party_level <= 20:
                raise ValueError("Party level must be between 1 and 20.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid party level: {str(e)}")
            self._append_to_combat_log(f"Error: Invalid party level: {str(e)}")
            return

        # Collect selected entity types and quantities
        entity_requests = {}
        total_quantity = 0
        for entity_type in self.entity_checkboxes:
            if self.entity_checkboxes[entity_type].isChecked():
                try:
                    qty = int(self.entity_quantity_inputs[entity_type].text())
                    if not 0 <= qty <= 10:
                        raise ValueError(f"Quantity for {entity_type} must be between 0 and 10.")
                    if qty > 0:
                        entity_requests[entity_type] = qty
                        total_quantity += qty
                except ValueError as e:
                    QMessageBox.warning(self, "Input Error", str(e))
                    self._append_to_combat_log(f"Error: {str(e)}")
                    return

        if not entity_requests:
            QMessageBox.warning(self, "Selection Error",
                                "Please select at least one entity type with a quantity greater than 0.")
            self._append_to_combat_log("Error: No entity types selected for placement.")
            return

        self._append_to_combat_log(f"Placing entities for party level {party_level}: {entity_requests}")

        # Cap total entities to available rooms
        max_entities = min(total_quantity, len(self.current_map_rooms))
        if max_entities < total_quantity:
            self._append_to_combat_log(
                f"Warning: Limited to {max_entities} entities due to {len(self.current_map_rooms)} rooms available.")
            total_quantity = max_entities

        # Fetch entities
        db_session = SessionLocal()
        available_entities = {}
        try:
            for entity_type, qty in entity_requests.items():
                if entity_type == "Monsters":
                    cr_max = max(1, party_level // 2)
                    query = db_session.query(Monster).filter(Monster.cr.in_([str(i) for i in range(1, cr_max + 1)]))
                    entities = query.limit(100).all()
                elif entity_type == "Magic Items":
                    rarities = ["Common", "Uncommon", "Rare"] if party_level < 10 else ["Common", "Uncommon", "Rare",
                                                                                        "Very Rare"]
                    query = db_session.query(MagicItem).filter(MagicItem.rarity.in_(rarities))
                    entities = query.limit(100).all()
                elif entity_type == "Armor":
                    query = db_session.query(Armor)
                    entities = query.limit(100).all()
                elif entity_type == "Weapons":
                    query = db_session.query(Weapon)
                    entities = query.limit(100).all()

                if not entities:
                    QMessageBox.warning(self, "Placement Error", f"No {entity_type} found in database.")
                    self._append_to_combat_log(f"Error: No {entity_type} found in database.")
                    return
                available_entities[entity_type] = entities
                self._append_to_combat_log(f"Found {len(entities)} {entity_type} for placement.")
        finally:
            db_session.close()

        # Clear existing entities
        self.active_map_entities.clear()
        self.map_with_entities_pixmap = self.map_pixmap.copy()

        # Initialize painter
        painter = QPainter(self.map_with_entities_pixmap)
        tile_size = 20
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Color coding for entities
        colors = {
            "Monsters": QColor("#FF0000"),  # Red
            "Magic Items": QColor("#00FF00"),  # Green
            "Armor": QColor("#0000FF"),  # Blue
            "Weapons": QColor("#FFFF00")  # Yellow
        }

        # Prepare placement list
        placement_list = []
        for entity_type, qty in entity_requests.items():
            placement_list.extend([(entity_type, qty) for _ in range(qty)])
        random.shuffle(placement_list)  # Randomize order to distribute entities evenly

        # Place entities
        placed = []
        used_rooms = set()
        for entity_type, _ in placement_list[:max_entities]:
            entities = available_entities[entity_type]
            entity = random.choice(entities)
            # Prefer unused rooms if available
            available_rooms = [r for i, r in enumerate(self.current_map_rooms) if i not in used_rooms]
            if not available_rooms:
                available_rooms = self.current_map_rooms
            room = random.choice(available_rooms)
            room_idx = self.current_map_rooms.index(room)
            used_rooms.add(room_idx)
            rx, ry, rw, rh = room

            # Ensure valid placement
            px = random.randint(rx + 1, rx + rw - 2)
            py = random.randint(ry + 1, ry + rh - 2)

            # Draw entity
            painter.setBrush(QBrush(colors[entity_type]))
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.drawEllipse(px * tile_size, py * tile_size, tile_size, tile_size)
            painter.setPen(QPen(QColor("#FFFFFF")))
            name = getattr(entity, "name", "X")[:1]
            painter.drawText(px * tile_size, py * tile_size, tile_size, tile_size,
                             Qt.AlignmentFlag.AlignCenter, name)

            # Store entity
            self.active_map_entities.append({
                "type": entity_type,
                "data": entity,
                "map_x": px,
                "map_y": py,
                "room": room
            })
            placed.append((entity_type, entity, px, py, room))

            # Add monsters to combat tracker
            if entity_type == "Monsters":
                try:
                    name = getattr(entity, "name", "Unknown Monster")
                    hp = getattr(entity, "hp", 10)  # Default HP if missing
                    self.add_combatant(name=name, hp=hp)
                    self._append_to_combat_log(f"Added {name} (HP: {hp}) to combat tracker.")
                except Exception as e:
                    self._append_to_combat_log(f"Error adding {name} to combat tracker: {str(e)}")

        painter.end()
        self._update_map_display()

        # Update description
        desc = self.map_description.toPlainText().split("\nPlaced Entities:\n")[0]  # Keep original map desc
        desc += "\nPlaced Entities:\n"
        for i, (entity_type, entity, x, y, room) in enumerate(placed, 1):
            name = getattr(entity, "name", "Unknown")
            desc += f"{i}. {entity_type}: {name} at ({x},{y}) in Room {self.current_map_rooms.index(room) + 1}\n"
        self.map_description.setPlainText(desc)

        self._append_to_combat_log(f"Placed {len(placed)} entities: {', '.join([f'{t}' for t, _, _, _, _ in placed])}")
        self.statusBar().showMessage(f"Placed {len(placed)} entities on the map.")
    def _on_generate_map_clicked(self):
        try:
            width = int(self.map_width_input.text())
            height = int(self.map_height_input.text())
            if not (5 <= width <= 50 and 5 <= height <= 50):
                raise ValueError("Width and height must be between 5 and 50.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return

        theme = self.map_theme_combo.currentText()
        self.generate_map(width, height, theme)

    def generate_map(self, width, height, theme):
        logging.debug(f"Generating map: {width}x{height}, theme: {theme}")
        self.current_map_rooms = []
        self.active_map_entities = []

        # Seed random
        random.seed()

        # Create grid: 0 = floor, 1 = wall
        grid = np.ones((width, height), dtype=int)

        # Generate rooms
        num_rooms = max(3, (width * height) // 100)
        rooms = []
        for _ in range(num_rooms * 2):  # Try more to ensure enough rooms
            rw = random.randint(3, min(6, width - 2))
            rh = random.randint(3, min(6, height - 2))
            rx = random.randint(1, width - rw - 1)
            ry = random.randint(1, height - rh - 1)

            # Check overlap
            overlap = False
            for ex, ey, ew, eh in rooms:
                if (rx < ex + ew + 1 and rx + rw + 1 > ex and
                        ry < ey + eh + 1 and ry + rh + 1 > ey):
                    overlap = True
                    break
            if overlap:
                continue

            # Carve room
            grid[rx:rx + rw, ry:ry + rh] = 0
            rooms.append((rx, ry, rw, rh))

        # Connect rooms
        for i in range(len(rooms) - 1):
            x1, y1, w1, h1 = rooms[i]
            x2, y2, w2, h2 = rooms[i + 1]
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

            # Horizontal corridor
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                grid[x, cy1] = 0
            # Vertical corridor
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                grid[cx2, y] = 0

        self.current_map_rooms = rooms

        # Render map
        tile_size = 16
        image = QImage(width * tile_size, height * tile_size, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.black)

        try:
            painter = QPainter(image)
            wall_brush = QBrush(QColor("#333333"))
            floor_brush = QBrush(QColor("#CCCCCC"))
            painter.setPen(QPen(Qt.PenStyle.NoPen))

            for x in range(width):
                for y in range(height):
                    painter.setBrush(wall_brush if grid[x, y] else floor_brush)
                    painter.drawRect(x * tile_size, y * tile_size, tile_size, tile_size)
        finally:
            if painter:
                painter.end()

        self.map_pixmap = QPixmap.fromImage(image)
        self.map_with_entities_pixmap = self.map_pixmap.copy() if self.map_pixmap else None
        self._update_map_display()

        # Generate description
        desc = f"{theme} Map ({width}x{height})\n"
        desc += f"Rooms: {len(rooms)}\n"
        desc += "Description:\n"
        if theme == "Dungeon":
            desc += "A dark, stone-walled dungeon with flickering torchlight.\n"
        elif theme == "Cave":
            desc += "A damp, stalactite-filled cave with echoing drips.\n"
        elif theme == "Castle":
            desc += "A grand castle hall with tapestries and suits of armor.\n"
        elif theme == "Forest":
            desc += "A dense forest clearing with overgrown paths.\n"
        desc += "\nRoom Details:\n"
        for i, (x, y, w, h) in enumerate(rooms, 1):
            desc += f"Room {i}: Position ({x},{y}), Size {w}x{h}\n"

        self.map_description.setPlainText(desc)
        self._append_to_combat_log(f"Generated {theme} map with {len(rooms)} rooms.")
        self.statusBar().showMessage(f"Generated {theme} map: {width}x{height}, {len(rooms)} rooms.")
        logging.debug(f"Map generated: {len(rooms)} rooms")

    def _place_entities(self):
        logging.debug("Starting _place_entities")
        if not self.current_map_rooms:
            QMessageBox.warning(self, "Placement Error", "No rooms available. Generate a map first.")
            self._append_to_combat_log("Error: No rooms available for entity placement.")
            logging.debug("No rooms available")
            return

        # Validate party level
        try:
            party_level = int(self.party_level_input.text())
            if not 1 <= party_level <= 20:
                raise ValueError("Party level must be between 1 and 20.")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid party level: {str(e)}")
            self._append_to_combat_log(f"Error: Invalid party level: {str(e)}")
            logging.debug(f"Invalid party level: {str(e)}")
            return

        # Collect selected entity types and quantities
        entity_requests = {}
        total_quantity = 0
        for entity_type in self.entity_checkboxes:
            if self.entity_checkboxes[entity_type].isChecked():
                try:
                    qty = int(self.entity_quantity_inputs[entity_type].text())
                    if not 0 <= qty <= 5:
                        raise ValueError(f"Quantity for {entity_type} must be between 0 and 5.")
                    if qty > 0:
                        entity_requests[entity_type] = qty
                        total_quantity += qty
                except ValueError as e:
                    QMessageBox.warning(self, "Input Error", str(e))
                    self._append_to_combat_log(f"Error: {str(e)}")
                    logging.debug(f"Invalid quantity for {entity_type}: {str(e)}")
                    return

        if not entity_requests:
            QMessageBox.warning(self, "Selection Error",
                                "Please select at least one entity type with a quantity greater than 0.")
            self._append_to_combat_log("Error: No entity types selected for placement.")
            logging.debug("No entity types selected")
            return

        self._append_to_combat_log(f"Placing entities for party level {party_level}: {entity_requests}")
        logging.debug(f"Entity requests: {entity_requests}, total: {total_quantity}")

        # Cap total entities
        max_entities = min(total_quantity, len(self.current_map_rooms), 20)
        if max_entities < total_quantity:
            self._append_to_combat_log(
                f"Warning: Limited to {max_entities} entities due to {len(self.current_map_rooms)} rooms available.")
            logging.debug(f"Limited to {max_entities} entities")

        # Fetch entities
        available_entities = {}
        db_session = SessionLocal()
        try:
            for entity_type, qty in entity_requests.items():
                logging.debug(f"Querying {entity_type}")
                if entity_type == "Monsters":
                    cr_max = max(1, party_level // 2)
                    query = db_session.query(Monster).filter(Monster.cr.in_([str(i) for i in range(1, cr_max + 1)]))
                    entities = query.limit(50).all()
                elif entity_type == "Magic Items":
                    rarities = ["Common", "Uncommon", "Rare"] if party_level < 10 else ["Common", "Uncommon", "Rare",
                                                                                        "Very Rare"]
                    query = db_session.query(MagicItem).filter(MagicItem.rarity.in_(rarities))
                    entities = query.limit(50).all()
                elif entity_type == "Armor":
                    query = db_session.query(Armor)
                    entities = query.limit(50).all()
                elif entity_type == "Weapons":
                    query = db_session.query(Weapon)
                    entities = query.limit(50).all()

                if not entities:
                    QMessageBox.warning(self, "Placement Error", f"No {entity_type} found in database.")
                    self._append_to_combat_log(f"Error: No {entity_type} found in database.")
                    logging.debug(f"No {entity_type} found")
                    return
                available_entities[entity_type] = entities
                self._append_to_combat_log(f"Found {len(entities)} {entity_type} for placement.")
                logging.debug(f"Found {len(entities)} {entity_type}")
        except Exception as e:
            self._append_to_combat_log(f"Database error: {str(e)}")
            logging.error(f"Database error: {str(e)}")
            return
        finally:
            db_session.close()
            logging.debug("Database session closed")

        # Clear existing entities
        self.active_map_entities.clear()
        logging.debug("Cleared active_map_entities")

        # Initialize pixmap
        try:
            self.map_with_entities_pixmap = self.map_pixmap.copy() if self.map_pixmap else None
            if self.map_with_entities_pixmap is None or self.map_with_entities_pixmap.isNull():
                raise ValueError("Invalid map pixmap")
        except Exception as e:
            QMessageBox.critical(self, "Pixmap Error", f"Failed to copy map pixmap: {str(e)}")
            self._append_to_combat_log(f"Error copying pixmap: {str(e)}")
            logging.error(f"Pixmap error: {str(e)}")
            return

        # Initialize painter
        painter = None
        try:
            painter = QPainter(self.map_with_entities_pixmap)
            tile_size = 16
            font = QFont("Arial", 8)
            painter.setFont(font)
        except Exception as e:
            QMessageBox.critical(self, "Painter Error", f"Failed to initialize QPainter: {str(e)}")
            self._append_to_combat_log(f"Error initializing QPainter: {str(e)}")
            logging.error(f"Painter error: {str(e)}")
            return

        # Color coding
        colors = {
            "Monsters": QColor("#FF0000"),
            "Magic Items": QColor("#00FF00"),
            "Armor": QColor("#0000FF"),
            "Weapons": QColor("#FFFF00")
        }

        # Prepare placement list
        placement_list = []
        for entity_type, qty in entity_requests.items():
            placement_list.extend([(entity_type, qty) for _ in range(qty)])
        random.shuffle(placement_list)
        logging.debug(f"Placement list: {len(placement_list)} items")

        # Place entities
        placed = []
        used_rooms = set()
        try:
            for i, (entity_type, _) in enumerate(placement_list[:max_entities]):
                logging.debug(f"Placing {entity_type} #{i + 1}")
                entities = available_entities[entity_type]
                entity = random.choice(entities)
                available_rooms = [r for j, r in enumerate(self.current_map_rooms) if j not in used_rooms]
                if not available_rooms:
                    available_rooms = self.current_map_rooms
                room = random.choice(available_rooms)
                room_idx = self.current_map_rooms.index(room)
                used_rooms.add(room_idx)
                rx, ry, rw, rh = room

                # Validate placement
                if rw <= 2 or rh <= 2:
                    self._append_to_combat_log(f"Skipping invalid room {room_idx + 1} with size {rw}x{rh}")
                    logging.debug(f"Invalid room {room_idx + 1}: {rw}x{rh}")
                    continue

                px = random.randint(rx + 1, rx + rw - 2)
                py = random.randint(ry + 1, ry + rh - 2)
                logging.debug(f"Position for {entity_type}: ({px}, {py})")

                # Draw entity
                painter.setBrush(QBrush(colors[entity_type]))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(px * tile_size, py * tile_size, tile_size, tile_size)
                painter.setPen(QPen(QColor("#FFFFFF")))
                name = getattr(entity, "name", "X")[:1]
                painter.drawText(px * tile_size, py * tile_size, tile_size, tile_size,
                                 Qt.AlignmentFlag.AlignCenter, name)
                logging.debug(f"Drew {entity_type} '{name}' at ({px}, {py})")

                # Store entity
                self.active_map_entities.append({
                    "type": entity_type,
                    "data": entity,
                    "map_x": px,
                    "map_y": py,
                    "room": room
                })
                placed.append((entity_type, entity, px, py, room))

                # Add monsters to combat tracker
                if entity_type == "Monsters":
                    try:
                        name = getattr(entity, "name", "Unknown Monster")
                        hp = getattr(entity, "hp", 10)
                        if not isinstance(hp, (int, float)):
                            hp = 10
                        self.add_combatant(name=name, hp=hp)
                        self._append_to_combat_log(f"Added {name} (HP: {hp}) to combat tracker.")
                        logging.debug(f"Added {name} (HP: {hp}) to combat tracker")
                    except Exception as e:
                        self._append_to_combat_log(f"Error adding {name} to combat tracker: {str(e)}")
                        logging.error(f"Combat tracker error for {name}: {str(e)}")

                QApplication.processEvents()
        except Exception as e:
            self._append_to_combat_log(f"Placement error: {str(e)}")
            logging.error(f"Placement error: {str(e)}")
        finally:
            if painter:
                painter.end()
            logging.debug("Painter ended")

        # Update description
        try:
            desc = self.map_description.toPlainText().split("\nPlaced Entities:")[0]
            desc += "\nPlaced Entities:\n"
            for i, (entity_type, entity, x, y, room) in enumerate(placed, 1):
                name = getattr(entity, "name", "Unknown")
                desc += f"{i}. {entity_type}: {name} at ({x},{y}) in Room {self.current_map_rooms.index(room) + 1}\n"
            self.map_description.setPlainText(desc)
        except Exception as e:
            self._append_to_combat_log(f"Description update error: {str(e)}")
            logging.error(f"Description error: {str(e)}")

        self._update_map_display()
        self._append_to_combat_log(f"Placed {len(placed)} entities")
        self.statusBar().showMessage(f"Placed {len(placed)} entities on the map.")
        logging.debug(f"Completed placement: {len(placed)} entities")
    def _save_map_as_image(self):
        if self.map_with_entities_pixmap and not self.map_with_entities_pixmap.isNull():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_filename = f"map_{timestamp}.png"
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Map Image", default_filename,
                                                       "PNG Images (*.png);;All Files (*)")
            if file_path:
                self.map_with_entities_pixmap.save(file_path)
                self.statusBar().showMessage(f"Map saved to '{os.path.basename(file_path)}'")
        else:
            QMessageBox.warning(self, "No Map", "No map has been generated to save.")

    def _update_map_display(self):
        if self.map_with_entities_pixmap and not self.map_with_entities_pixmap.isNull():
            self.map_label.setPixmap(self.map_with_entities_pixmap.scaled(
                self.map_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def create_combat_tracker_tab(self):
        """Create and initialize the combat tracker tab."""
        if "combat_tracker" in self.tab_initialized:
            logging.debug("Combat tracker tab already initialized")
            return

        combat_tab = QWidget()
        layout = QVBoxLayout()

        # Round counter
        self.round_counter_label = QLabel(f"Round: {self.current_round}")
        layout.addWidget(self.round_counter_label)

        # Combat log
        self.combat_log = QTextEdit()
        self.combat_log.setReadOnly(True)
        self.combat_log.setMinimumHeight(100)
        layout.addWidget(self.combat_log)

        # Initiative table
        self.initiative_table = QTableWidget()
        self.initiative_table.setColumnCount(3)
        self.initiative_table.setHorizontalHeaderLabels(["Name", "Initiative", "HP"])
        self.initiative_table.setSortingEnabled(True)
        layout.addWidget(self.initiative_table)

        # Ailment table
        self.ailment_table = QTableWidget()
        self.ailment_table.setColumnCount(4)
        self.ailment_table.setHorizontalHeaderLabels(["Target", "Ailment", "Duration", "Source"])
        self.ailment_table.setSortingEnabled(True)
        layout.addWidget(self.ailment_table)

        # Input fields for combatants
        input_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Combatant Name")
        self.hp_input = QLineEdit()
        self.hp_input.setPlaceholderText("HP")
        self.hp_input.setValidator(QIntValidator(0, 9999))
        self.initiative_input = QLineEdit()
        self.initiative_input.setPlaceholderText("Initiative (optional)")
        self.initiative_input.setValidator(QIntValidator(0, 99))
        input_layout.addWidget(self.name_input)
        input_layout.addWidget(self.hp_input)
        input_layout.addWidget(self.initiative_input)
        layout.addLayout(input_layout)

        # Input fields for ailments
        ailment_input_layout = QHBoxLayout()
        self.ailment_target_input = QLineEdit()
        self.ailment_target_input.setPlaceholderText("Target Name")
        self.ailment_name_input = QLineEdit()
        self.ailment_name_input.setPlaceholderText("Ailment Name")
        self.ailment_duration_input = QLineEdit()
        self.ailment_duration_input.setPlaceholderText("Duration (turns)")
        self.ailment_duration_input.setValidator(QIntValidator(1, 99))
        ailment_input_layout.addWidget(self.ailment_target_input)
        ailment_input_layout.addWidget(self.ailment_name_input)
        ailment_input_layout.addWidget(self.ailment_duration_input)
        layout.addLayout(ailment_input_layout)

        # Buttons
        button_layout = QHBoxLayout()
        add_combatant_button = QPushButton("Add Combatant")
        add_combatant_button.clicked.connect(self._add_combatant_from_input)
        button_layout.addWidget(add_combatant_button)

        add_ailment_button = QPushButton("Add Ailment")
        add_ailment_button.clicked.connect(self._add_ailment)
        button_layout.addWidget(add_ailment_button)

        remove_combatant_button = QPushButton("Remove Selected Combatant")
        remove_combatant_button.clicked.connect(self._remove_selected_combatant)
        button_layout.addWidget(remove_combatant_button)

        remove_ailment_button = QPushButton("Remove Selected Ailment")
        remove_ailment_button.clicked.connect(self._remove_selected_ailment)
        button_layout.addWidget(remove_ailment_button)

        next_turn_button = QPushButton("Next Turn")
        next_turn_button.clicked.connect(self._next_turn)
        button_layout.addWidget(next_turn_button)

        sort_initiative_button = QPushButton("Sort Initiative")
        sort_initiative_button.clicked.connect(self._sort_initiative)
        button_layout.addWidget(sort_initiative_button)

        damage_button = QPushButton("Deal Damage")
        damage_button.clicked.connect(lambda: self._update_combatant_hp("damage"))
        button_layout.addWidget(damage_button)

        heal_button = QPushButton("Heal")
        heal_button.clicked.connect(lambda: self._update_combatant_hp("heal"))
        button_layout.addWidget(heal_button)

        layout.addLayout(button_layout)

        combat_tab.setLayout(layout)
        self.central_widget.addTab(combat_tab, "Combat Tracker")
        self.tab_initialized["combat_tracker"] = True
        logging.debug("Combat tracker tab created")

    def create_session_manager_tab(self):
        session_tab = QWidget()
        session_tab.setObjectName("session_manager_tab")
        main_layout = QVBoxLayout(session_tab)
        playback_group = QGroupBox("Playback & Analysis")
        playback_main_layout = QVBoxLayout()
        playback_controls_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Session", self)
        self.load_button.clicked.connect(self.load_session)
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_recording)
        self.transcribe_button = QPushButton("Transcribe...", self)
        self.transcribe_button.clicked.connect(self.run_transcription)
        playback_controls_layout.addWidget(self.load_button)
        playback_controls_layout.addWidget(self.play_button)
        playback_controls_layout.addWidget(self.transcribe_button)
        playback_controls_layout.addStretch()
        self.playback_label = QLabel("PLAY: 00:00 / 00:00", self)
        playback_controls_layout.addWidget(self.playback_label)
        playback_main_layout.addLayout(playback_controls_layout)
        self.playback_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.playback_slider.setTracking(True)
        self.playback_slider.sliderMoved.connect(self.seek_audio)
        self.playback_slider.sliderReleased.connect(self.seek_audio)
        playback_main_layout.addWidget(self.playback_slider)
        playback_group.setLayout(playback_main_layout)
        main_layout.addWidget(playback_group)
        recording_group = QGroupBox("Live Recording")
        recording_layout = QVBoxLayout()
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Input Device:", self))
        self.audio_device_combo = QComboBox(self)
        self._populate_audio_devices()
        self.audio_device_combo.currentIndexChanged.connect(self._on_audio_device_changed)  # Connect signal
        device_layout.addWidget(self.audio_device_combo)
        refresh_device_button = QPushButton("Refresh Devices", self)  # Add refresh button
        refresh_device_button.clicked.connect(self._populate_audio_devices)
        device_layout.addWidget(refresh_device_button)
        recording_layout.addLayout(device_layout)
        record_controls_layout = QHBoxLayout()
        self.record_button = QPushButton("Record", self)
        self.record_button.clicked.connect(self.start_recording)
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_or_resume_recording)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_recording)
        record_controls_layout.addWidget(self.record_button)
        record_controls_layout.addWidget(self.pause_button)
        record_controls_layout.addWidget(self.stop_button)
        record_controls_layout.addStretch()
        self.mic_level_bar = QProgressBar(self)
        self.mic_level_bar.setRange(0, 100)
        self.mic_level_bar.setValue(0)
        self.mic_level_bar.setTextVisible(False)
        self.mic_level_bar.setFixedSize(150, 15)
        record_controls_layout.addWidget(self.mic_level_bar)
        self.timer_label = QLabel("REC: 00:00", self)
        record_controls_layout.addWidget(self.timer_label)
        recording_layout.addLayout(record_controls_layout)
        recording_group.setLayout(recording_layout)
        main_layout.addWidget(recording_group)
        notes_group = QGroupBox("Session Notes & Timestamps")
        notes_main_layout = QVBoxLayout()
        content_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        timestamp_widget = QWidget()
        timestamp_layout = QVBoxLayout(timestamp_widget)
        timestamp_layout.addWidget(QLabel("Timestamps", self))
        self.timestamp_list = QListWidget(self)
        self.timestamp_list.itemClicked.connect(self.on_timestamp_item_clicked)
        timestamp_layout.addWidget(self.timestamp_list)
        mark_note_layout = QHBoxLayout()
        self.timestamp_note_input = QLineEdit(self)
        self.timestamp_note_input.setPlaceholderText("Brief note for new timestamp...")
        mark_note_layout.addWidget(self.timestamp_note_input)
        mark_timestamp_button = QPushButton("Mark", self)
        mark_timestamp_button.clicked.connect(self.add_timestamp)
        mark_note_layout.addWidget(mark_timestamp_button)
        timestamp_layout.addLayout(mark_note_layout)
        content_splitter.addWidget(timestamp_widget)
        self.general_notes_editor = QTextEdit(self)
        self.general_notes_editor.setPlaceholderText("General session notes and observations...")
        content_splitter.addWidget(self.general_notes_editor)
        content_splitter.setSizes([400, 600])
        notes_main_layout.addWidget(content_splitter)
        notes_group.setLayout(notes_main_layout)
        main_layout.addWidget(notes_group)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.timer_label.setVisible(False)
        self.playback_label.setVisible(False)
        self.mic_level_bar.setVisible(False)
        self.playback_slider.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self.central_widget.addTab(session_tab, "Session Manager")

    def _save_api_key(self, api_key):
        """Saves the Gemini API key to a config file."""
        config = {'gemini_api_key': api_key}
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f)
        except IOError as e:
            print(f"Warning: Could not save API key to '{self.CONFIG_FILE}': {e}")
            self.statusBar().showMessage("Warning: Could not save API key.")

    def _load_api_key(self):
        """Loads the Gemini API key from the config file."""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('gemini_api_key')
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error reading API key from '{self.CONFIG_FILE}': {e}")
                self.statusBar().showMessage("Error loading API key from config.")
        return None

    def _get_gemini_api_key(self):
        """Loads the Gemini API key from config or prompts the user if not found/invalid."""
        api_key = self._load_api_key()

        if api_key:
            try:
                genai.configure(api_key=api_key.strip())
                # Attempt a dummy call to verify the key
                test_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                test_model.generate_content("test")
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                self.statusBar().showMessage("Gemini API key loaded from config and validated.")
                self.set_ai_buttons_enabled(True)
                return  # Successfully loaded and validated
            except Exception as e:
                QMessageBox.warning(self, "Saved API Key Error",
                                    f"The saved Gemini API Key is invalid or there was a connection error: {e}\n"
                                    "Please enter a new key.")
                api_key = None  # Invalidate the loaded key

        # Loop until a valid key is entered or the user cancels
        while self.gemini_model is None:
            new_api_key, ok = QInputDialog.getText(
                self,
                "Gemini API Key",
                "Enter your Google Gemini API Key (Go to https://aistudio.google.com/ to get one).\n"
                "This will be saved for future sessions. Leave blank and click OK to skip AI features.",
                QLineEdit.EchoMode.Normal,
                ""
            )

            if ok and new_api_key:
                try:
                    genai.configure(api_key=new_api_key.strip())
                    # Attempt a dummy call to verify the new key
                    test_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    test_model.generate_content("test")
                    # Set the main model
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                    self.statusBar().showMessage("Gemini API key validated, configured, and saved.")
                    self._save_api_key(new_api_key.strip())  # Save the new valid key
                    self.set_ai_buttons_enabled(True)
                    break  # Break the loop as we have a valid model
                except Exception as e:
                    QMessageBox.critical(self, "API Key Error",
                                         f"Invalid Gemini API Key or connection error: {e}\n"
                                         "AI features will remain disabled. Please try again.")
                    self.gemini_model = None  # Ensure it's still None to re-prompt
            else:
                # User clicked Cancel or left the field blank
                QMessageBox.information(self, "API Key Skipped",
                                        "Gemini API key was not provided. AI features will be disabled.")
                self.set_ai_buttons_enabled(False)
                break  # Exit the loop

    def set_ai_buttons_enabled(self, is_enabled):
        if self.gemini_model is None:
            is_enabled = False

        ai_tab = self.findChild(QWidget, "ai_assistant_tab")
        if ai_tab:
            for button in ai_tab.findChildren(QPushButton):
                button.setEnabled(is_enabled)
            if hasattr(self, 'ai_query_input') and self.ai_query_input:
                self.ai_query_input.setEnabled(is_enabled)

    def run_gemini_with_full_prompt(self, full_prompt):
        if self.gemini_model is None:
            QMessageBox.critical(self, "API Error", "Gemini model is not configured. Please enter a valid API key.")
            self.set_ai_buttons_enabled(False)
            return
        self.set_ai_buttons_enabled(False)
        self.statusBar().showMessage("Analyzing text with Gemini...")
        self.gemini_worker = GeminiWorker(model=self.gemini_model, prompt=full_prompt)
        self.gemini_worker.generation_finished.connect(self.on_generation_finished)
        self.gemini_worker.finished.connect(self.gemini_worker.deleteLater)
        self.gemini_worker.start()

    def on_generation_finished(self, result_text, error_text):
        if error_text:
            QMessageBox.critical(self, "API Error", f"An error occurred:\n{error_text}")
        else:
            dialog = ResponseDialog(result_text, self)
            dialog.timestamp_clicked.connect(self.play_recording)
            dialog.exec()
        self.set_ai_buttons_enabled(True)
        self.statusBar().showMessage("Ready")

    def create_ai_assistant_tab(self):
        ai_tab = QWidget()
        ai_tab.setObjectName("ai_assistant_tab")
        layout = QVBoxLayout(ai_tab)
        layout.addWidget(QLabel("<h2>Gemini AI Assistant</h2>", self))

        query_group = QGroupBox("Ask the AI")
        query_layout = QVBoxLayout()
        self.ai_query_input = QLineEdit(self)
        self.ai_query_input.setPlaceholderText(
            "Type your question or request for the AI (e.g., 'Give me a short description of a goblin ambush')...")
        query_layout.addWidget(self.ai_query_input)

        query_button_layout = QHBoxLayout()
        send_query_button = QPushButton("Send Query", self)
        send_query_button.clicked.connect(self._on_ai_query_clicked)
        query_button_layout.addWidget(send_query_button)
        query_button_layout.addStretch()
        query_layout.addLayout(query_button_layout)
        query_group.setLayout(query_layout)
        layout.addWidget(query_group)

        analysis_group = QGroupBox("Analyze Pasted Text")
        analysis_layout = QVBoxLayout()
        analysis_layout.addWidget(QLabel("Paste Transcript or Notes to Analyze", self))
        self.analysis_text_box = QTextEdit(self)
        self.analysis_text_box.setPlaceholderText("Paste a session transcript or notes here...")
        analysis_layout.addWidget(self.analysis_text_box)

        analysis_button_layout = QHBoxLayout()
        summarize_button = QPushButton("Generate Session Summary", self)
        summarize_button.clicked.connect(self.run_session_summary)
        analysis_button_layout.addWidget(summarize_button)
        extract_button = QPushButton("Extract Key Information", self)
        extract_button.clicked.connect(self.run_information_extraction)
        analysis_button_layout.addWidget(extract_button)
        analysis_button_layout.addStretch()
        analysis_layout.addLayout(analysis_button_layout)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        predefined_prompts_group = QGroupBox("Quick Prompts")
        predefined_prompts_layout = QVBoxLayout()

        row1_buttons = QHBoxLayout()
        gen_encounter_button = QPushButton("Random Encounter", self)
        gen_encounter_button.clicked.connect(self._on_gen_random_encounter_clicked)
        row1_buttons.addWidget(gen_encounter_button)

        describe_npc_button = QPushButton("Describe an NPC", self)
        describe_npc_button.clicked.connect(self._on_describe_npc_clicked)
        row1_buttons.addWidget(describe_npc_button)
        row1_buttons.addStretch()
        predefined_prompts_layout.addLayout(row1_buttons)

        row2_buttons = QHBoxLayout()
        create_plothook_button = QPushButton("Create Plot Hook", self)
        create_plothook_button.clicked.connect(self._on_create_plothook_clicked)
        row2_buttons.addWidget(create_plothook_button)

        gen_dungeon_room_button = QPushButton("Dungeon Room Idea", self)
        gen_dungeon_room_button.clicked.connect(self._on_gen_dungeon_room_clicked)
        row2_buttons.addWidget(gen_dungeon_room_button)
        row2_buttons.addStretch()
        predefined_prompts_layout.addLayout(row2_buttons)

        predefined_prompts_layout.addStretch()
        predefined_prompts_group.setLayout(predefined_prompts_layout)
        layout.addWidget(predefined_prompts_group)

        layout.addStretch()
        self.central_widget.addTab(ai_tab, "AI Assistant")

    def _on_ai_query_clicked(self):
        query_text = self.ai_query_input.text().strip()
        if not query_text:
            QMessageBox.warning(self, "Input Error", "Please enter a query for the AI.")
            return

        full_prompt = (f"Act as a helpful assistant for a Dungeon Master. Respond to the following request:\n\n"
                       f"---\n\n{query_text}")
        self.run_gemini_with_full_prompt(full_prompt)
        self.ai_query_input.clear()

    def _on_gen_random_encounter_clicked(self):
        prompt_suffix, ok = QInputDialog.getText(self, "Random Encounter",
                                                 "Describe the environment or context for the encounter (optional):",
                                                 QLineEdit.EchoMode.Normal, "a forest road")
        if ok:
            full_prompt = (f"Act as a helpful assistant for a Dungeon Master. Generate a random D&D 5e encounter "
                           f"idea. Include a brief description of the scene, potential creatures, a challenge rating (CR) "
                           f"suggestion, and a minor plot hook or complication. Context: {prompt_suffix.strip() if prompt_suffix.strip() else 'generic fantasy setting'}")
            self.run_gemini_with_full_prompt(full_prompt)

    def _on_describe_npc_clicked(self):
        prompt_suffix, ok = QInputDialog.getText(self, "Describe an NPC",
                                                 "Provide some keywords or context for the NPC (e.g., 'grumpy dwarf innkeeper', 'mysterious elven ranger'):",
                                                 QLineEdit.EchoMode.Normal, "kind halfling merchant")
        if ok:
            full_prompt = (
                f"Act as a helpful assistant for a Dungeon Master. Create a brief description for a D&D NPC. "
                f"Include their appearance, personality quirk, a secret or goal, and a potential hook for players. "
                f"Keywords: {prompt_suffix.strip() if prompt_suffix.strip() else 'generic fantasy NPC'}")
            self.run_gemini_with_full_prompt(full_prompt)

    def _on_create_plothook_clicked(self):
        prompt_suffix, ok = QInputDialog.getText(self, "Create Plot Hook",
                                                 "Provide a theme or keywords for the plot hook (e.g., 'ancient ruins', 'missing villagers', 'political intrigue'):",
                                                 QLineEdit.EchoMode.Normal, "a strange artifact")
        if ok:
            full_prompt = (f"Act as a helpful assistant for a Dungeon Master. Generate a D&D plot hook idea. "
                           f"Describe the initial situation, the inciting incident, and what the players might need to do. "
                           f"Theme/Keywords: {prompt_suffix.strip() if prompt_suffix.strip() else 'generic fantasy adventure'}")
            self.run_gemini_with_full_prompt(full_prompt)

    def _on_gen_dungeon_room_clicked(self):
        prompt_suffix, ok = QInputDialog.getText(self, "Dungeon Room Idea",
                                                 "Describe the type of dungeon or room (e.g., 'goblin cave', 'ancient library', 'magical trap room'):",
                                                 QLineEdit.EchoMode.Normal, "a forgotten crypt")
        if ok:
            full_prompt = (f"Act as a helpful assistant for a Dungeon Master. Describe a D&D dungeon room. "
                           f"Include its appearance, notable features, potential traps or puzzles, and any monsters or treasures. "
                           f"Context: {prompt_suffix.strip() if prompt_suffix.strip() else 'a generic dungeon'}")
            self.run_gemini_with_full_prompt(full_prompt)

    def _populate_audio_devices(self):
        try:
            devices = sd.query_devices()
            self.audio_device_combo.clear()  # Clear existing items
            input_devices = [d for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            if not input_devices:
                self.audio_device_combo.addItem("No input devices found")
                self.audio_device_combo.setEnabled(False)
                self.statusBar().showMessage("No audio input devices detected.")
                logging.debug("No input devices found")
                return

            self.audio_device_combo.setEnabled(True)
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.audio_device_combo.addItem(f"{device['name']} (ID: {i})", i)
            # Set default device to system default if available
            default_input = sd.default.device[0]  # Get default input device index
            for index in range(self.audio_device_combo.count()):
                if self.audio_device_combo.itemData(index) == default_input:
                    self.audio_device_combo.setCurrentIndex(index)
                    break
            self.statusBar().showMessage(f"Audio input: {self.audio_device_combo.currentText()}")
            logging.debug(f"Populated {self.audio_device_combo.count()} input devices")
        except Exception as e:
            self.audio_device_combo.addItem("Error querying devices")
            self.audio_device_combo.setEnabled(False)
            self.statusBar().showMessage(f"Error querying audio devices: {e}")
            logging.error(f"Error querying audio devices: {e}")

    def _on_audio_device_changed(self, index):
        if index == -1 or not self.audio_device_combo.isEnabled():
            return
        device_id = self.audio_device_combo.itemData(index)
        if device_id is None:
            self.statusBar().showMessage("Invalid audio device selected")
            logging.debug("Invalid audio device selected")
            return
        try:
            device_info = sd.query_devices(device_id)
            if device_info['max_input_channels'] <= 0:
                raise ValueError("Selected device has no input channels")
            # Validate sample rate compatibility
            supported = sd.check_input_settings(device=device_id, samplerate=self.sample_rate, channels=1)
            self.statusBar().showMessage(f"Selected audio input: {self.audio_device_combo.currentText()}")
            logging.debug(f"Audio device changed to ID {device_id}: {device_info['name']}")
        except Exception as e:
            self.statusBar().showMessage(f"Selected device may be incompatible: {e}")
            self.audio_device_combo.setCurrentIndex(0)  # Revert to first device
            logging.warning(f"Invalid device {device_id}: {e}")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.recording_state == "recording":
            self.recorded_frames.append(indata.copy())
            self.last_peak_level = np.abs(indata).max()

    def _update_timer_display(self):
        if self.recording_state == "recording":
            self.elapsed_seconds += 1
            minutes, seconds = divmod(self.elapsed_seconds, 60)
            self.timer_label.setText(f"REC: {minutes:02d}:{seconds:02d}")

        if self.playback_stream and self.playback_stream.active and self.loaded_audio_data is not None:
            # Calculate current time from current_frame
            current_seconds = self.current_frame // self.sample_rate
            current_minutes, current_seconds = divmod(current_seconds, 60)

            # Calculate total time from loaded_audio_data
            total_seconds = len(self.loaded_audio_data) // self.sample_rate
            total_minutes, total_seconds_rem = divmod(total_seconds, 60)

            self.playback_label.setText(
                f"PLAY: {current_minutes:02d}:{current_seconds:02d} / {total_minutes:02d}:{total_seconds_rem:02d}"
            )

            # Update slider position if not being dragged
            if not self.playback_slider.isSliderDown():
                self.playback_slider.blockSignals(True)  # Prevent recursive signal emission
                self.playback_slider.setValue(current_seconds)
                self.playback_slider.blockSignals(False)

    def _update_mic_level(self):
        level = int(self.last_peak_level * 100)
        self.mic_level_bar.setValue(level)

    def add_timestamp(self):
        if self.recording_state != "recording":
            QMessageBox.warning(self, "Timestamp Error", "Can only add a timestamp while recording.")
            return
        note_text = self.timestamp_note_input.text().strip() or "Mark"
        time_in_seconds = self.elapsed_seconds
        timestamp_data = (time_in_seconds, note_text)
        self.session_timestamps.append(timestamp_data)
        minutes, seconds = divmod(time_in_seconds, 60)
        display_text = f"[{minutes:02d}:{seconds:02d}] {note_text}"
        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.ItemDataRole.UserRole, time_in_seconds)
        self.timestamp_list.addItem(list_item)
        self.timestamp_note_input.clear()

    def start_recording(self):
        self.stop_playback()
        self.session_timestamps.clear()
        self.timestamp_list.clear()
        self.elapsed_seconds = 0
        self.last_peak_level = 0
        self._update_timer_display()
        self._update_mic_level()
        self.recorded_frames.clear()
        selected_device_index = self.audio_device_combo.currentData()
        if selected_device_index is None:
            QMessageBox.critical(self, "Audio Error", "No valid input device selected or found.")
            logging.error("No valid input device selected")
            return

        try:
            # Validate device settings
            device_info = sd.query_devices(selected_device_index)
            sd.check_input_settings(device=selected_device_index, samplerate=self.sample_rate, channels=1)
            self.recording_state = "recording"
            self.statusBar().showMessage(f"Recording from: {self.audio_device_combo.currentText()}")
            self.recording_timer.start(1000)
            self.level_check_timer.start(100)
            self.timer_label.setVisible(True)
            self.mic_level_bar.setVisible(True)
            self.record_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.load_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.transcribe_button.setEnabled(False)
            self.audio_device_combo.setEnabled(False)
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                device=selected_device_index,
                channels=1,
                callback=self._audio_callback
            )
            self.audio_stream.start()
            logging.debug(f"Started recording with device ID {selected_device_index}: {device_info['name']}")
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Could not start recording: {e}")
            self.statusBar().showMessage(f"Recording failed: {e}")
            self.stop_recording()
            logging.error(f"Recording error with device {selected_device_index}: {e}")
    def stop_recording(self):
        if self.recording_state == "stopped":
            return
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self.recording_timer.stop()
        self.level_check_timer.stop()
        self.recording_state = "stopped"
        self.timer_label.setVisible(False)
        self.mic_level_bar.setVisible(False)
        self.mic_level_bar.setValue(0)
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.load_button.setEnabled(True)
        if not self.recorded_frames:
            self.statusBar().showMessage("Recording stopped. No audio data.")
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"session_{timestamp}"
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Session Audio", default_filename, "WAV Files (*.wav)")
        if filePath:
            wav_path = filePath
            base_path = os.path.splitext(wav_path)[0]
            timestamp_path = f"{base_path}.timestamps.txt"
            recording = np.concatenate(self.recorded_frames, axis=0)
            sf.write(wav_path, recording, self.sample_rate)
            try:
                with open(timestamp_path, 'w', encoding='utf-8') as f:
                    for seconds, note in self.session_timestamps:
                        f.write(f"{seconds}|{note}\n")
                self.statusBar().showMessage(f"Session saved to '{os.path.basename(base_path)}'")
                self.current_playback_filepath = wav_path
                self.transcribe_button.setEnabled(True)
                self.play_button.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(self, "Timestamp Save Error", f"Could not save timestamps file: {e}")
        else:
            self.statusBar().showMessage("Save cancelled.")

    def pause_or_resume_recording(self):
        if self.recording_state == "recording":
            self.recording_state = "paused"
            self.level_check_timer.stop()
            self.recording_timer.stop()
            self.mic_level_bar.setValue(0)
            self.pause_button.setText("Resume")
            self.statusBar().showMessage("Recording paused.")
        elif self.recording_state == "paused":
            self.recording_state = "recording"
            self.level_check_timer.start(100)
            self.recording_timer.start(1000)
            self.pause_button.setText("Pause")
            self.statusBar().showMessage("Recording...")

    def load_session(self):
        self.stop_playback()
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Session Audio", "", "WAV Files (*.wav)")
        if not filePath:
            return
        try:
            self.loaded_audio_data, file_samplerate = sf.read(filePath, dtype='float32')
            if file_samplerate != self.sample_rate:
                QMessageBox.warning(self, "Sample Rate Mismatch",
                                    f"Audio file has a different sample rate ({file_samplerate}) than project ({self.sample_rate}). Playback quality may be affected.")
            if self.loaded_audio_data.ndim == 1:
                self.loaded_audio_data = self.loaded_audio_data.reshape(-1, 1)
            self.current_playback_filepath = filePath
            self.play_button.setEnabled(True)
            self.transcribe_button.setEnabled(True)
            self.playback_label.setVisible(True)
            total_seconds = len(self.loaded_audio_data) // self.sample_rate
            self.playback_slider.setRange(0, total_seconds)
            self.playback_slider.setEnabled(True)
            self._update_timer_display()  # Update UI immediately
            # Load timestamps
            self.timestamp_list.clear()
            base_path = os.path.splitext(filePath)[0]
            timestamp_path = f"{base_path}.timestamps.txt"
            if os.path.exists(timestamp_path):
                try:
                    with open(timestamp_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('|', 1)
                            if len(parts) == 2:
                                seconds, note = int(parts[0]), parts[1]
                                minutes, sec = divmod(seconds, 60)
                                display_text = f"[{minutes:02d}:{sec:02d}] {note}"
                                list_item = QListWidgetItem(display_text)
                                list_item.setData(Qt.ItemDataRole.UserRole, seconds)
                                self.timestamp_list.addItem(list_item)
                except Exception as e:
                    QMessageBox.warning(self, "Timestamp Load Error", f"Could not load timestamps file: {e}")
            self.play_recording()  # Start playback immediately after loading
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Could not load audio file: {e}")

    def on_timestamp_item_clicked(self, item):
        start_time = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(start_time, int):
            self.play_recording(start_time=start_time)

    def _playback_callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunk_end = self.current_frame + frames
        remaining_frames = len(self.loaded_audio_data) - self.current_frame
        if remaining_frames >= frames:
            outdata[:] = self.loaded_audio_data[self.current_frame:chunk_end]
            self.current_frame += frames
        else:
            outdata[:remaining_frames] = self.loaded_audio_data[self.current_frame:]
            outdata[remaining_frames:] = 0
            raise sd.CallbackStop

    def play_recording(self, start_time=0):
        if self.loaded_audio_data is None:
            QMessageBox.warning(self, "Playback Error", "No audio session loaded. Please use 'Load Session'.")
            logging.debug("Playback attempted with no loaded audio data")
            return

        # Stop any existing playback
        self.stop_playback()

        # Set the starting frame based on start_time
        self.current_frame = int(start_time * self.sample_rate)
        self.playback_slider.setValue(start_time)

        try:
            self.playback_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.loaded_audio_data.shape[1],
                callback=self._playback_callback,
                finished_callback=self.stop_playback
            )
            self.playback_stream.start()
            self.recording_timer.start(1000)  # Update every second
            self.play_button.setText("Stop Playback")
            self.play_button.clicked.disconnect()
            self.play_button.clicked.connect(self.stop_playback)
            self.playback_label.setVisible(True)
            self.playback_slider.setEnabled(True)
            logging.debug(f"Started playback from {start_time} seconds")
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", f"Could not start playback stream: {e}")
            self.stop_playback()
            logging.error(f"Playback error: {e}")

    def stop_playback(self):
        stream = self.playback_stream
        if stream:
            self.playback_stream = None
            stream.stop()
            stream.close()
        self.recording_timer.stop()
        self.playback_label.setVisible(False)
        self.playback_slider.setValue(0)
        self.playback_slider.setEnabled(False)
        self.play_button.setText("Play")
        try:
            self.play_button.clicked.disconnect()
        except TypeError:
            pass
        self.play_button.clicked.connect(self.play_recording)

    def seek_audio(self):
        if self.loaded_audio_data is None:
            logging.debug("Seek attempted with no loaded audio data")
            return
        seek_time = self.playback_slider.value()
        logging.debug(f"Seeking to {seek_time} seconds")
        if self.playback_stream and self.playback_stream.active:
            # Update current_frame directly instead of restarting stream
            self.current_frame = int(seek_time * self.sample_rate)
            self._update_timer_display()
        else:
            # If no stream is active, start playback from seek_time
            self.play_recording(start_time=seek_time)

    def run_transcription(self):
        if not self.gemini_model:
            QMessageBox.critical(self, "API Error", "Gemini model is not configured.")
            return
        if not self.current_playback_filepath:
            QMessageBox.warning(self, "Transcription Error", "Please load a session file first.")
            return
        self.set_ai_buttons_enabled(False)
        self.transcribe_button.setEnabled(False)
        self.transcriber_worker = AudioTranscriberWorker(model=self.gemini_model,
                                                         file_path=self.current_playback_filepath)
        self.transcriber_worker.transcription_finished.connect(self.on_transcription_finished)
        self.transcriber_worker.finished.connect(self.transcriber_worker.deleteLater)
        self.transcriber_worker.start()

    def on_transcription_finished(self, transcript, status_or_error):
        if transcript:
            dialog = ResponseDialog(transcript, self)
            dialog.timestamp_clicked.connect(self.play_recording)
            dialog.exec()
            self.analysis_text_box.setPlainText(transcript)
        else:
            QMessageBox.warning(self, "Transcription Status", status_or_error)
        self.set_ai_buttons_enabled(True)
        self.transcribe_button.setEnabled(True)

    def run_session_summary(self):
        source_text = self.analysis_text_box.toPlainText()
        if not source_text:
            QMessageBox.warning(self, "Input Error", "Please paste a transcript or notes to summarize.")
            return
        prompt = (f"Act as a helpful assistant for a Dungeon Master. Read the following D&D session notes/transcript "
                  f"and provide a concise summary of the key events, important player decisions, and major outcomes."
                  f"\n\n---\n\n{source_text}")
        self.run_gemini_with_full_prompt(prompt)

    def run_information_extraction(self):
        source_text = self.analysis_text_box.toPlainText()
        if not source_text:
            QMessageBox.warning(self, "Input Error", "Please paste a transcript or notes to extract information from.")
            return
        prompt = (f"Act as a helpful assistant for a Dungeon Master. Read the following D&D session notes/transcript "
                  f"and extract the following information. If a category has no information, write 'None'.\n\n"
                  f"- **Key NPCs Mentioned:**\n"
                  f"- **Locations Visited or Described:**\n"
                  f"- **Quests Started or Advanced:**\n"
                  f"- **Unique Items or Loot Found:**\n\n"
                  f"---\n\n{source_text}")
        self.run_gemini_with_full_prompt(prompt)

    def _refresh_npc_tab(self):
        """Rebuilds and populates the entire NPC tab."""
        # Define the headers for the table summary
        headers = ["Name", "Type", "Location", "Status", "Race", "Class"]

        # Main widget and layout
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Control buttons
        controls_group = QGroupBox("NPC Management")
        controls_layout = QHBoxLayout()
        add_button = QPushButton("Add New NPC")
        add_button.clicked.connect(self._on_add_npc_clicked)
        edit_button = QPushButton("View/Edit Selected NPC")
        edit_button.clicked.connect(self._on_edit_npc_clicked)
        remove_button = QPushButton("Remove Selected NPC")
        remove_button.clicked.connect(self._on_remove_npc_clicked)
        controls_layout.addWidget(add_button)
        controls_layout.addWidget(edit_button)
        controls_layout.addWidget(remove_button)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Create and configure the table
        self.npc_table = QTableWidget(0, len(headers))
        self.npc_table.setHorizontalHeaderLabels(headers)
        self.npc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.npc_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.npc_table.setSortingEnabled(True)
        self.npc_table.itemDoubleClicked.connect(self._on_edit_npc_clicked)
        layout.addWidget(self.npc_table)

        # Fetch data from DB and populate table
        db_session = SessionLocal()
        try:
            npcs = db_session.query(NPC).order_by(NPC.name).all()
            self.npc_table.setRowCount(len(npcs))
            for row_num, npc in enumerate(npcs):
                # Store the npc id in the first item for easy retrieval
                name_item = QTableWidgetItem(npc.name)
                name_item.setData(Qt.ItemDataRole.UserRole, npc.id)

                self.npc_table.setItem(row_num, 0, name_item)
                self.npc_table.setItem(row_num, 1, QTableWidgetItem(npc.npc_type))
                self.npc_table.setItem(row_num, 2, QTableWidgetItem(npc.location))
                self.npc_table.setItem(row_num, 3, QTableWidgetItem(npc.status))
                self.npc_table.setItem(row_num, 4, QTableWidgetItem(npc.race))
                self.npc_table.setItem(row_num, 5, QTableWidgetItem(npc.npc_class))
        finally:
            db_session.close()

        # Replace the old tab content with the new, fully-featured widget
        self.npc_tab_content = widget
        self._replace_entity_tab("NPCs", self.npc_tab_content, 5)  # Assuming NPC is the 6th tab (index 5)

    def _on_add_npc_clicked(self):
        """Handles the 'Add New NPC' button click."""
        dialog = AddEditNPCDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "Input Error", "NPC name cannot be empty.")
                return

            new_npc = NPC(**data)
            db_session = SessionLocal()
            try:
                db_session.add(new_npc)
                db_session.commit()
                self.statusBar().showMessage(f"NPC '{data['name']}' saved successfully.")
                self._refresh_npc_tab()
            except Exception as e:
                db_session.rollback()
                QMessageBox.critical(self, "Database Error",
                                     f"Could not save NPC. Does an NPC with this name already exist?\n\nDetails: {e}")
            finally:
                db_session.close()

    def _on_edit_npc_clicked(self):
        """Handles editing the selected NPC."""
        selected_items = self.npc_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select an NPC to view or edit.")
            return

        # Get the ID we stored in the name item's UserRole
        npc_id = selected_items[0].data(Qt.ItemDataRole.UserRole)

        db_session = SessionLocal()
        try:
            npc_to_edit = db_session.query(NPC).filter(NPC.id == npc_id).first()
            if not npc_to_edit:
                QMessageBox.critical(self, "Error",
                                     "Could not find the selected NPC in the database. It may have been deleted.")
                self._refresh_npc_tab()
                return

            dialog = AddEditNPCDialog(npc=npc_to_edit, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                data = dialog.get_data()
                if not data["name"]:
                    QMessageBox.warning(self, "Input Error", "NPC name cannot be empty.")
                    return

                # Update the existing NPC object with new data
                for key, value in data.items():
                    setattr(npc_to_edit, key, value)

                db_session.commit()
                self.statusBar().showMessage(f"NPC '{data['name']}' updated successfully.")
                self._refresh_npc_tab()
        except Exception as e:
            db_session.rollback()
            QMessageBox.critical(self, "Database Error", f"Could not update NPC.\n\nDetails: {e}")
        finally:
            db_session.close()

    def _on_remove_npc_clicked(self):
        """Handles removing the selected NPC."""
        selected_items = self.npc_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select an NPC to remove.")
            return

        npc_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        npc_name = selected_items[0].text()

        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to permanently delete '{npc_name}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        db_session = SessionLocal()
        try:
            npc_to_delete = db_session.query(NPC).filter(NPC.id == npc_id).first()
            if npc_to_delete:
                db_session.delete(npc_to_delete)
                db_session.commit()
                self.statusBar().showMessage(f"NPC '{npc_name}' has been deleted.")
                self._refresh_npc_tab()
        except Exception as e:
            db_session.rollback()
            QMessageBox.critical(self, "Database Error", f"Could not delete NPC.\n\nDetails: {e}")
        finally:
            db_session.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # --- Apply Custom UI Styling ---
    # 1. Load the custom pixel font
    pixel_font = ui_styles.load_custom_font()
    app.setFont(pixel_font) # Set it as the default font for the whole app

    # 2. Load and apply the QSS stylesheet
    stylesheet = ui_styles.get_stylesheet()
    app.setStyleSheet(stylesheet)
    # --- End of UI Styling ---

    dm_assistant = DungeonMasterAssistant()
    dm_assistant.show()
    sys.exit(app.exec())
