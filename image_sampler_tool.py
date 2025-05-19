import os
import sys
import shutil
import random
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List

from PyQt5 import QtWidgets, QtCore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
FUN_MESSAGES = [
    "Crunching numbers...",
    "Sampling some greatness...",
    "Copying with style...",
]

@dataclass
class FolderEntry:
    path: str
    percent: int = 0
    blank: int = 5
    basic: int = 15
    detailed: int = 30
    structured: int = 50

    def caption_total(self) -> int:
        return self.blank + self.basic + self.detailed + self.structured

class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, entries: List[FolderEntry], dest: str, total_count: int, mode: str):
        super().__init__()
        self.entries = entries
        self.dest = dest
        self.total_count = total_count
        self.mode = mode
        log_file = os.path.join(dest, "sampler.log")
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")

    def run(self):
        total_images = sum(len(self._list_images(e.path)) for e in self.entries)
        processed = 0
        self.message.emit(random.choice(FUN_MESSAGES))
        logging.info("Starting sampling to %s", self.dest)
        for entry in self.entries:
            images = self._list_images(entry.path)
            if self.mode == "total":
                quota = int(self.total_count * entry.percent / 100)
            else:
                quota = int(len(images) * entry.percent / 100)
            if quota > len(images):
                quota = len(images)
                self.message.emit(f"{entry.path}: using max available {quota} images")
            logging.info("%s: selecting %d images", entry.path, quota)
            selected = random.sample(images, quota)
            for img in selected:
                base = os.path.basename(img)
                name, _ = os.path.splitext(base)
                dest_img = os.path.join(self.dest, base)
                dest_caption = os.path.join(self.dest, name + ".txt")
                shutil.copy2(img, dest_img)
                style = self._copy_caption(entry, name, dest_caption)
                logging.info("%s -> %s (%s)", img, dest_img, style)
                processed += 1
                if total_images:
                    self.progress.emit(int(processed / total_images * 100))
                if processed % 10 == 0:
                    self.message.emit(random.choice(FUN_MESSAGES))
        self.finished.emit()

    def _list_images(self, folder: str) -> List[str]:
        files = []
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                files.append(os.path.join(folder, f))
        return files

    def _copy_caption(self, entry: FolderEntry, name: str, dest_caption: str) -> str:
        style = random.choices(
            ["blank", "basic", "detailed", "structured"],
            weights=[entry.blank, entry.basic, entry.detailed, entry.structured],
        )[0]
        if style == "blank":
            open(dest_caption, "w").close()
            return style
        caption_file = os.path.join(entry.path, f"{name}.{style}.txt")
        if not os.path.exists(caption_file):
            caption_file = os.path.join(entry.path, f"{name}.txt")
        if os.path.exists(caption_file):
            shutil.copy2(caption_file, dest_caption)
        else:
            open(dest_caption, "w").close()
        return style


class FolderDropWidget(QtWidgets.QWidget):
    folderDropped = QtCore.pyqtSignal(str)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.folderDropped.emit(path)


class FolderWidget(QtWidgets.QWidget):
    def __init__(self, entry: FolderEntry, parent=None):
        super().__init__(parent)
        self.entry = entry
        layout = QtWidgets.QGridLayout(self)
        self.path_label = QtWidgets.QLabel(entry.path)
        self.percent_spin = QtWidgets.QSpinBox()
        self.percent_spin.setRange(0, 100)
        self.percent_spin.setValue(entry.percent)
        self.blank_spin = QtWidgets.QSpinBox()
        self.basic_spin = QtWidgets.QSpinBox()
        self.detailed_spin = QtWidgets.QSpinBox()
        self.structured_spin = QtWidgets.QSpinBox()
        for s, v in zip(
            [self.blank_spin, self.basic_spin, self.detailed_spin, self.structured_spin],
            [entry.blank, entry.basic, entry.detailed, entry.structured],
        ):
            s.setRange(0, 100)
            s.setValue(v)
        layout.addWidget(self.path_label, 0, 0, 1, 5)
        layout.addWidget(QtWidgets.QLabel("Folder %"), 1, 0)
        layout.addWidget(self.percent_spin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Blank"), 1, 2)
        layout.addWidget(self.blank_spin, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Basic"), 2, 0)
        layout.addWidget(self.basic_spin, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Detailed"), 2, 2)
        layout.addWidget(self.detailed_spin, 2, 3)
        layout.addWidget(QtWidgets.QLabel("Structured"), 3, 0)
        layout.addWidget(self.structured_spin, 3, 1)

    def update_entry(self):
        self.entry.percent = self.percent_spin.value()
        self.entry.blank = self.blank_spin.value()
        self.entry.basic = self.basic_spin.value()
        self.entry.detailed = self.detailed_spin.value()
        self.entry.structured = self.structured_spin.value()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Sampler")
        self.entries: List[FolderEntry] = []
        self.settings = QtCore.QSettings("sampler", "dataset")
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Classic", "Total Count"])
        layout.addWidget(self.mode_combo)

        self.total_spin = QtWidgets.QSpinBox()
        self.total_spin.setMaximum(1_000_000)
        self.total_spin.setValue(1000)
        layout.addWidget(self.total_spin)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.list_widget = FolderDropWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.list_widget)
        self.list_widget.folderDropped.connect(self.add_folder)
        self.scroll.setWidget(self.list_widget)
        layout.addWidget(self.scroll)

        btn_layout = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add Folder")
        self.remove_btn = QtWidgets.QPushButton("Remove Folder")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        layout.addLayout(btn_layout)

        dest_layout = QtWidgets.QHBoxLayout()
        self.dest_edit = QtWidgets.QLineEdit()
        self.dest_btn = QtWidgets.QPushButton("Browse")
        dest_layout.addWidget(QtWidgets.QLabel("Destination"))
        dest_layout.addWidget(self.dest_edit)
        dest_layout.addWidget(self.dest_btn)
        layout.addLayout(dest_layout)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.start_btn = QtWidgets.QPushButton("Start")
        layout.addWidget(self.start_btn)

        self.add_btn.clicked.connect(self.add_folder)
        self.remove_btn.clicked.connect(self.remove_folder)
        self.dest_btn.clicked.connect(self.browse_dest)
        self.start_btn.clicked.connect(self.start_sampling)

    def add_folder(self, folder=None):
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        entry = FolderEntry(path=folder)
        widget = FolderWidget(entry)
        self.entries.append(entry)
        self.list_layout.addWidget(widget)

    def remove_folder(self):
        if self.list_layout.count() == 0:
            return
        item = self.list_layout.takeAt(self.list_layout.count() - 1)
        if item:
            widget = item.widget()
            widget.setParent(None)
            self.entries.pop()

    def browse_dest(self):
        dest = QtWidgets.QFileDialog.getExistingDirectory(self, "Destination Folder")
        if dest:
            self.dest_edit.setText(dest)

    def start_sampling(self):
        for i in range(self.list_layout.count()):
            widget = self.list_layout.itemAt(i).widget()
            widget.update_entry()
        if self.mode_combo.currentIndex() == 1 and self.entries:
            max_entry = max(self.entries, key=lambda e: e.percent)
            for i, entry in enumerate(self.entries):
                w = self.list_layout.itemAt(i).widget()
                color = "green" if entry is max_entry else ""
                w.path_label.setStyleSheet(f"color:{color}")
        else:
            for i in range(self.list_layout.count()):
                self.list_layout.itemAt(i).widget().path_label.setStyleSheet("")
        if self.mode_combo.currentIndex() == 1:
            if sum(e.percent for e in self.entries) != 100:
                QtWidgets.QMessageBox.warning(self, "Error", "Folder percentages must sum to 100")
                return
        for e in self.entries:
            if e.caption_total() != 100:
                QtWidgets.QMessageBox.warning(self, "Error", f"Caption percentages for {e.path} must sum to 100")
                return
        dest = self.dest_edit.text()
        if not dest:
            QtWidgets.QMessageBox.warning(self, "Error", "Destination folder required")
            return
        os.makedirs(dest, exist_ok=True)
        if os.listdir(dest):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(dest, timestamp)
            os.makedirs(dest, exist_ok=True)
        mode = "total" if self.mode_combo.currentIndex() == 1 else "classic"
        total = self.total_spin.value()
        self.worker = Worker(self.entries, dest, total, mode)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.message.connect(self.log_box.append)
        self.worker.finished.connect(lambda: QtWidgets.QMessageBox.information(self, "Done", "Sampling complete"))
        self.worker.start()

    def _load_settings(self):
        dest = self.settings.value("dest", "")
        self.dest_edit.setText(dest)
        folders_json = self.settings.value("folders", "[]")
        try:
            folders = json.loads(folders_json)
        except Exception:
            folders = []
        for data in folders:
            entry = FolderEntry(**data)
            widget = FolderWidget(entry)
            self.entries.append(entry)
            self.list_layout.addWidget(widget)

    def closeEvent(self, event):
        self.settings.setValue("dest", self.dest_edit.text())
        data = [e.__dict__ for e in self.entries]
        self.settings.setValue("folders", json.dumps(data))
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
